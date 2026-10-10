"""
Batch extractor for processing folders of audio files.
"""
import copy
import csv
import json
import os
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np

from afx_audio.extract_all import FEATURE_MAP, extract_all_features
from afx_audio.io.io import load_audio
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


def _write_parquet(path: str, results: dict[str, Any]) -> None:
    import pandas as pd

    keys, columns, rows = _table(results)
    df = pd.DataFrame(rows, index=pd.Index(keys, name='file'), columns=columns)
    df.to_parquet(path)


def _process_one_file(
    fpath: str, fkey: str, config: dict[str, Any], custom: dict | None = None
):
    """
    Worker: load one audio file and extract features.

    Module-level (picklable) so it can run in a worker process. Returns
    (fkey, feats_as_lists, error_message) with exactly one of the last two
    set to None.
    """
    if custom:
        restore_custom_features(custom)
    try:
        signal, sr = load_audio(fpath, sr=config['sample_rate'])
        feats = extract_all_features(signal, sr, config)
        if isinstance(feats, np.ndarray):
            # aggregation='stack' returns a single stacked vector
            return fkey, {'stack': feats.tolist()}, None
        return fkey, {k: np.asarray(v).tolist() for k, v in feats.items()}, None
    except Exception as e:
        return fkey, None, str(e)


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
            Workers use the platform's default start method; with spawn
            (Windows, macOS) and forkserver (Linux from Python 3.14) a
            script must call this under ``if __name__ == '__main__':``.
        sample_rate: Override the config's sample_rate (audio is resampled
            to this rate on load).
        verbose: Print per-file progress to stdout.
        coverage: Add a <feature>_valid_frac column per feature (fraction
            of frames with a finite value). None keeps the config's setting.

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
    """
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
    total = len(all_files)

    # Key by path relative to the root folder so same-named files in
    # different subfolders don't overwrite each other.
    # Forward slashes on every OS, so outputs written on Windows and Linux match.
    keys = [os.path.relpath(fpath, folder_path).replace(os.sep, '/') for fpath in all_files]

    if n_jobs is not None and n_jobs != 1 and total > 1:
        import concurrent.futures
        import multiprocessing

        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, total)
        if verbose:
            print(f'Processing {total} file(s) with {n_jobs} workers')
        # Registered custom features do not exist in fresh worker processes
        custom = custom_features()
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = {
                executor.submit(_process_one_file, fpath, fkey, config, custom): fkey
                for fpath, fkey in zip(all_files, keys, strict=False)
            }
            done = 0
            for future in concurrent.futures.as_completed(futures):
                fkey, feats, err = future.result()
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
            fkey, feats, err = _process_one_file(fpath, fkey, config)
            if err is not None:
                errors[fkey] = err
            else:
                results[fkey] = feats

    if errors:
        errors_path = f'{output_path}.errors.json'
        with open(errors_path, 'w', encoding='utf-8') as f:
            json.dump(errors, f, indent=2)
        warnings.warn(f'{len(errors)} file(s) failed; details in {errors_path}', stacklevel=2)

    if total == 0:
        warnings.warn(
            f'no audio files with extensions {tuple(file_exts)} found under {folder_path}',
            stacklevel=2,
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
        _write_parquet(output_path, results)

    out = dict(results)
    if errors:
        out['errors'] = errors
    return out
