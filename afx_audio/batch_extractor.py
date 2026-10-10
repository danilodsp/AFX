"""
Batch extractor for processing folders of audio files.
"""
import copy
import csv
import json
import multiprocessing
import os
import sys
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import soundfile as sf

from afx_audio._warnings import AFXWarning
from afx_audio.dataset_stats import STATS_SUFFIX, DatasetStats, write_stats
from afx_audio.extract_all import FEATURE_MAP, extract_all_features
from afx_audio.io.io import load_audio
from afx_audio.provenance import PROVENANCE_KEY, build_provenance, canonical_json, file_sha256, write_sidecar
from afx_audio.registry import custom_features, restore_custom_features
from afx_audio.utils.batch_utils import flatten_features
from afx_audio.utils.config_loader import load_config, unknown_feature_error, validate_config

_DEFAULT_EXTS = ('.wav', '.mp3', '.flac', '.ogg')


def _flatten_to_columns(feats: dict[str, list]) -> dict[str, float]:
    """One scalar column per value, named as in afx_audio.utils.batch_utils.feature_names."""
    names, values = flatten_features(feats)
    return dict(zip(names, values.tolist(), strict=True))


def _table(results: dict[str, Any]) -> tuple[list[str], list[str], list[dict[str, float]]]:
    """Row keys, the union of columns in first-seen order, and the rows."""
    rows = [_flatten_to_columns(feats) for feats in results.values()]
    columns = list(dict.fromkeys(c for row in rows for c in row))
    if any(len(row) != len(columns) for row in rows):
        warnings.warn(
            'Files produced different numbers of values, so some cells are empty. With '
            "aggregation='stack' every clip must have the same length; use a pooling aggregation otherwise.",
            AFXWarning,
            stacklevel=3,
        )
    return list(results), columns, rows


def _json_safe(value):
    """NaN and infinity become null: strict JSON parsers reject the bare NaN that json.dump writes."""
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


FORMATS = ('npy', 'json', 'csv', 'parquet')


def _output_format(output_path: str, save_format: str | None) -> str:
    if save_format is None:
        ext = os.path.splitext(output_path)[1].lower().lstrip('.')
        if ext not in FORMATS:
            raise ValueError(
                f'Cannot infer the output format from {output_path!r}; use one of the extensions '
                f'{", ".join("." + f for f in FORMATS)} or give the format (save_format, afx --format)'
            )
        return ext
    if save_format not in FORMATS:
        raise ValueError(f'Unknown save_format {save_format!r}. Valid: {FORMATS}')
    return save_format


def _check_output(output_path: str, save_format: str) -> None:
    """Fail before extracting anything if the output cannot be written."""
    folder = os.path.dirname(os.path.abspath(output_path))
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'Output folder does not exist: {folder}')
    if save_format == 'parquet':
        try:
            import pandas  # noqa: F401
            import pyarrow  # noqa: F401
        except ImportError as e:
            raise ImportError('Parquet output requires pandas and pyarrow: pip install "afx-audio[parquet]"') from e


def _write_csv(path: str, results: dict[str, Any]) -> None:
    keys, columns, rows = _table(results)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['file', *columns])
        for key, row in zip(keys, rows, strict=True):
            writer.writerow([key, *(repr(row[c]) if c in row else '' for c in columns)])


def _write_parquet(path: str, results: dict[str, Any], provenance: dict[str, Any]) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    keys, columns, rows = _table(results)
    df = pd.DataFrame(rows, index=pd.Index(keys, name='file'), columns=columns)
    table = pa.Table.from_pandas(df)
    metadata = {**(table.schema.metadata or {}), PROVENANCE_KEY.encode(): canonical_json(provenance).encode()}
    pq.write_table(table.replace_schema_metadata(metadata), path)


def _dataset_stats(results: dict[str, Any]) -> dict[str, Any]:
    """Statistics over the same flattened columns as the csv and parquet outputs, rows in key order."""
    stats = DatasetStats()
    for feats in results.values():
        stats.update(*flatten_features(feats))
    return stats.to_dict()


def _mp_context() -> multiprocessing.context.BaseContext:
    """
    Start method for the worker processes: the one set with multiprocessing.set_start_method, else the
    platform default, with forkserver in place of fork.

    fork copies the parent with its locks but not its threads, so a lock held by another thread (an executor
    feeder thread, a BLAS pool) deadlocks the child; Python 3.12+ warns about it and 3.14 dropped fork as the
    Linux default.
    """
    methods = multiprocessing.get_all_start_methods()
    method = multiprocessing.get_start_method(allow_none=True) or methods[0]
    if method == 'fork' and 'forkserver' in methods:
        method = 'forkserver'
    return multiprocessing.get_context(method)


def _check_importable(custom: dict) -> None:
    """Workers find a custom feature by module and name, which fails for a function defined in a notebook or REPL."""
    interactive = not getattr(sys.modules.get('__main__'), '__file__', None)
    for name, (func, _, _) in custom.items():
        if interactive and getattr(func, '__module__', None) == '__main__':
            raise ValueError(
                f'Custom feature {name!r} is defined interactively (notebook or REPL), so worker processes cannot '
                'import it. Define it in a module and import it from there, or use n_jobs=1.'
            )


def _process_one_file(
    fpath: str, fkey: str, config: dict[str, Any], custom: dict | None = None, hash_file: bool = False
):
    """
    Worker: load one audio file and extract features.

    Module-level (picklable) so it can run in a worker process. Returns
    (fkey, feats_as_lists, error_message, file_info) with exactly one of
    feats and error set to None; file_info holds the native sample rate
    and, with hash_file, the sha256 of the file bytes, for every file that
    loaded.
    """
    if custom:
        restore_custom_features(custom)
    info: dict[str, Any] = {}
    try:
        signal, sr = load_audio(fpath, sr=config['sample_rate'])
        info['native_sr'] = sf.info(fpath).samplerate
        if hash_file:
            # Hashed in the worker so it overlaps with other files' extraction
            info['sha256'] = file_sha256(fpath)
        feats = extract_all_features(signal, sr, config)
        if isinstance(feats, np.ndarray):
            # aggregation='stack' returns a single stacked vector
            return fkey, {'stack': feats.tolist()}, None, info
        return fkey, {k: np.asarray(v).tolist() for k, v in feats.items()}, None, info
    except Exception as e:
        return fkey, None, str(e), info


def extract_features_from_folder(
    folder_path: str,
    config_path: str | dict[str, Any],
    output_path: str,
    file_exts: Sequence[str] | None = None,
    save_format: str | None = None,
    features: list[str] | None = None,
    aggregation: str | None = None,
    flatten: bool | None = None,
    normalize: str | None = None,
    n_jobs: int | None = None,
    sample_rate: int | None = None,
    verbose: bool = True,
    coverage: bool | None = None,
    hash_files: bool = True,
    dataset_stats: bool = False,
) -> dict[str, Any]:
    """
    Extract features from all audio files in a folder (recursively).

    Args:
        folder_path: Root folder to search for audio files
        config_path: Path to a config JSON, or an already loaded config
            dict (for example from ``afx_audio.presets.load_preset``). A dict is
            copied, not modified.
        output_path: Path to save output (npy, json, csv, or parquet). Its
            folder must exist.
        file_exts: File extensions to include (default: wav/mp3/flac/ogg)
        save_format: Output format ('npy', 'json', 'csv', 'parquet'); None
            infers it from the output_path extension.
        features: Optional list of feature names to extract. Names missing
            from the config are added with default parameters; unknown
            names raise ValueError.
        aggregation: Aggregation method to use (validated)
        flatten: Whether to flatten features after aggregation. None keeps
            the config file's setting (default True if unset there).
        normalize: Normalization method ('zscore', 'minmax', or None)
        n_jobs: Number of worker processes for parallel extraction.
            None or 1 = serial; 0 or negative = all available cores.
            Results are identical to serial extraction (same ordering).
            Workers never fork: they start with forkserver where fork is
            the default (Linux before Python 3.14), otherwise with the
            method set by ``multiprocessing.set_start_method`` or the
            platform default (spawn on Windows and macOS). Either way
            each worker imports the calling script, so a script must call
            this under ``if __name__ == '__main__':``, and a registered
            custom feature must be a top-level function of a module or
            script (not a lambda, nor defined in a notebook).
        sample_rate: Override the config's sample_rate (audio is resampled
            to this rate on load).
        verbose: Print per-file progress to stdout.
        coverage: Add a <feature>_valid_frac column per feature (fraction
            of frames with a finite value). None keeps the config's setting.
        hash_files: Record the sha256 of every audio file in the
            provenance record. Costs one extra read per file.
        dataset_stats: Also write per-column count, mean, variance
            (ddof=0), min, max and NaN count over the flattened feature
            vectors to '<output_path>.stats.json'; see
            ``afx_audio.load_stats``.

    Returns:
        The results dict, keyed by path relative to folder_path. Files that
        failed are collected under the 'errors' entry of the returned dict
        and written to '<output_path>.errors.json' (never mixed into the
        main output).

    Notes:
        - 'npy' saves a pickled dict; load it back with
          ``np.load(path, allow_pickle=True).item()``. Prefer 'json' for
          interchange with other tools; NaN is written as null there.
        - 'csv' and 'parquet' write one row per file with scalar columns;
          vector features are expanded to ``name_0, name_1, ...`` columns.
          'parquet' requires pandas and pyarrow (``pip install "afx-audio[parquet]"``)
          and is the recommended format for dataset-scale extraction.
        - Every output records how it was made (afx-audio, numpy and scipy
          versions, the validated config, its hash and preset id, sample rate and
          resampling, file hashes): in the parquet schema metadata under
          the key 'afx', otherwise in '<output_path>.provenance.json'.
          Read it with ``afx_audio.read_provenance``.
    """
    return _extract_folder(
        folder_path, config_path, output_path, file_exts=file_exts, save_format=save_format, features=features,
        aggregation=aggregation, flatten=flatten, normalize=normalize, n_jobs=n_jobs, sample_rate=sample_rate,
        verbose=verbose, coverage=coverage, hash_files=hash_files, dataset_stats=dataset_stats,
    )[0]


def _extract_folder(
    folder_path: str,
    config_path: str | dict[str, Any],
    output_path: str,
    *,
    file_exts: Sequence[str] | None,
    save_format: str | None,
    features: list[str] | None,
    aggregation: str | None,
    flatten: bool | None,
    normalize: str | None,
    n_jobs: int | None,
    sample_rate: int | None,
    verbose: bool,
    coverage: bool | None,
    hash_files: bool,
    dataset_stats: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """extract_features_from_folder, also returning the provenance record it wrote."""
    if isinstance(config_path, dict):
        config = validate_config(copy.deepcopy(config_path))
    else:
        config = load_config(config_path)
    if sample_rate is not None:
        config['sample_rate'] = sample_rate

    # Override config options if requested
    if features is not None:
        for name in features:
            if name not in FEATURE_MAP:
                raise unknown_feature_error(name)
        config['features'] = {name: config['features'].get(name, {}) for name in features}
    if aggregation is not None:
        config['aggregation'] = aggregation
    if flatten is not None:
        config['flatten'] = flatten
    if normalize is not None:
        config['normalize'] = normalize
    if coverage is not None:
        config['coverage'] = coverage
    config = validate_config(config)
    save_format = _output_format(output_path, save_format)
    _check_output(output_path, save_format)

    if file_exts is None:
        file_exts = _DEFAULT_EXTS

    # Collect files first so we can show progress
    all_files = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in file_exts):
                all_files.append(os.path.join(root, fname))
    all_files.sort()

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}
    file_info: dict[str, dict[str, Any]] = {}
    total = len(all_files)

    # Key by path relative to the root folder so same-named files in
    # different subfolders don't overwrite each other.
    # Forward slashes on every OS, so outputs written on Windows and Linux match.
    keys = [os.path.relpath(fpath, folder_path).replace(os.sep, '/') for fpath in all_files]

    if n_jobs is not None and n_jobs != 1 and total > 1:
        import concurrent.futures

        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, total)
        if verbose:
            print(f'Processing {total} file(s) with {n_jobs} workers')
        # Registered custom features do not exist in fresh worker processes
        custom = {name: entry for name, entry in custom_features().items() if name in config['features']}
        _check_importable(custom)
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs, mp_context=_mp_context()) as executor:
            futures = {
                executor.submit(_process_one_file, fpath, fkey, config, custom, hash_files): fkey
                for fpath, fkey in zip(all_files, keys, strict=False)
            }
            done = 0
            for future in concurrent.futures.as_completed(futures):
                fkey, feats, err, info = future.result()
                file_info[fkey] = info
                done += 1
                if verbose:
                    print(f'[{done}/{total}] Processed {fkey}')
                if err is not None:
                    errors[fkey] = err
                else:
                    results[fkey] = feats
        # Restore deterministic (sorted) ordering regardless of
        # completion order, so parallel output == serial output.
        results = {k: results[k] for k in keys if k in results}
    else:
        for idx, (fpath, fkey) in enumerate(zip(all_files, keys, strict=False), 1):
            if verbose:
                print(f'[{idx}/{total}] Processing {fkey}')
            fkey, feats, err, info = _process_one_file(fpath, fkey, config, hash_file=hash_files)
            file_info[fkey] = info
            if err is not None:
                errors[fkey] = err
            else:
                results[fkey] = feats
    file_info = {k: file_info[k] for k in keys if file_info.get(k)}

    if errors:
        errors_path = f'{output_path}.errors.json'
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        warnings.warn(f'{len(errors)} file(s) failed; details in {errors_path}', AFXWarning, stacklevel=2)

    if total == 0:
        warnings.warn(
            f'no audio files with extensions {tuple(file_exts)} found under {folder_path}',
            AFXWarning,
            stacklevel=2,
        )

    provenance = build_provenance(
        config,
        output_format=save_format,
        native_rates={k: v['native_sr'] for k, v in file_info.items()},
        file_hashes={k: v['sha256'] for k, v in file_info.items()} if hash_files else None,
        sample_rate_overridden=sample_rate is not None,
        n_files=total,
        n_errors=len(errors),
        custom_features=sorted(name for name in custom_features() if name in config['features']),
    )

    if save_format == 'npy':
        # A file object, so numpy does not append .npy to other extensions
        with open(output_path, 'wb') as f:
            np.save(f, np.asarray(results, dtype=object))
    elif save_format == 'json':
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(_json_safe(results), f, indent=2, allow_nan=False)
    elif save_format == 'csv':
        _write_csv(output_path, results)
    else:
        _write_parquet(output_path, results, provenance)
    if save_format != 'parquet':
        write_sidecar(output_path, provenance)
    if dataset_stats:
        write_stats(f'{output_path}{STATS_SUFFIX}', _dataset_stats(results),
                    afx_version=provenance['afx_version'], config_hash=provenance['config_hash'])

    out = dict(results)
    if errors:
        out['errors'] = errors
    return out, provenance
