"""
Batch extractor for processing folders of audio files.
"""
import os
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List, Optional, Sequence

from AFX.io.io import load_audio
from AFX.utils.config_loader import load_config
from AFX.extract_all import extract_all_features, FEATURE_MAP

_DEFAULT_EXTS = ('.wav', '.mp3', '.flac', '.ogg')

_VALID_AGGREGATIONS = (
    'mean', 'std', 'summary', 'advanced_summary', 'skew', 'kurtosis',
    'sliding_window', 'stack',
)


def _flatten_to_columns(feats: Dict[str, list]) -> Dict[str, float]:
    """
    Flatten a {feature: list-of-values} dict into scalar columns:
    single-value features keep their name, multi-value features expand
    to name_0, name_1, ...
    """
    row = {}
    for name, values in feats.items():
        arr = np.asarray(values).ravel()
        if arr.size == 1:
            row[name] = float(arr[0])
        else:
            for i, v in enumerate(arr):
                row[f'{name}_{i}'] = float(v)
    return row


def _process_one_file(
    fpath: str, fkey: str, config: Dict[str, Any]
):
    """
    Worker: load one audio file and extract features.

    Module-level (picklable) so it can run in a worker process. Returns
    (fkey, feats_as_lists, error_message) with exactly one of the last two
    set to None.
    """
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
    config_path: str,
    output_path: str,
    file_exts: Optional[Sequence[str]] = None,
    save_format: str = 'npy',
    features: Optional[List[str]] = None,
    aggregation: Optional[str] = None,
    flatten: Optional[bool] = None,
    normalize: Optional[str] = None,
    n_jobs: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract features from all audio files in a folder (recursively).

    Args:
        folder_path: Root folder to search for audio files
        config_path: Path to config.json
        output_path: Path to save output (npy, json, csv, or parquet)
        file_exts: File extensions to include (default: wav/mp3/flac/ogg)
        save_format: Output format ('npy', 'json', 'csv', 'parquet')
        features: Optional list of feature names to extract. Unknown names
            raise ValueError.
        aggregation: Aggregation method to use (validated)
        flatten: Whether to flatten features after aggregation. None keeps
            the config file's setting (default True if unset there).
        normalize: Normalization method ('zscore', 'minmax', or None)
        n_jobs: Number of worker processes for parallel extraction.
            None or 1 = serial; 0 or negative = all available cores.
            Results are identical to serial extraction (same ordering).

    Returns:
        The results dict, keyed by path relative to folder_path. Files that
        failed are collected under the 'errors' entry of the returned dict
        and written to '<output_path>.errors.json' (never mixed into the
        main output).

    Notes:
        - 'npy' saves a pickled dict; load it back with
          ``np.load(path, allow_pickle=True).item()``. Prefer 'json' for
          interchange with other tools.
        - 'csv' and 'parquet' write one row per file with scalar columns;
          vector features are expanded to ``name_0, name_1, ...`` columns.
          'parquet' requires ``pyarrow`` (``pip install "AFX[parquet]"``)
          and is the recommended format for dataset-scale extraction.
    """
    config = load_config(config_path)

    # Override config options if requested
    if features is not None:
        known = set(config['features'])
        unknown = [f for f in features if f not in known]
        if unknown:
            raise ValueError(
                f'Unknown feature(s) requested: {unknown}. '
                f'Features in config: {sorted(known)}. '
                f'All available extractors: {sorted(FEATURE_MAP)}'
            )
        config['features'] = {
            k: v for k, v in config['features'].items() if k in features
        }
    if aggregation is not None:
        if aggregation not in _VALID_AGGREGATIONS:
            raise ValueError(
                f'Unknown aggregation: {aggregation!r}. '
                f'Valid: {_VALID_AGGREGATIONS}'
            )
        config['aggregation'] = aggregation
    if flatten is not None:
        config['flatten'] = flatten
    if normalize is not None:
        config['normalize'] = normalize

    if file_exts is None:
        file_exts = _DEFAULT_EXTS

    # Collect files first so we can show progress
    all_files = []
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if any(fname.lower().endswith(ext) for ext in file_exts):
                all_files.append(os.path.join(root, fname))
    all_files.sort()

    results: Dict[str, Any] = {}
    errors: Dict[str, str] = {}
    total = len(all_files)

    # Key by path relative to the root folder so same-named files in
    # different subfolders don't overwrite each other.
    keys = [os.path.relpath(fpath, folder_path) for fpath in all_files]

    if n_jobs is not None and n_jobs != 1 and total > 1:
        import concurrent.futures
        import multiprocessing

        if n_jobs <= 0:
            n_jobs = multiprocessing.cpu_count()
        n_jobs = min(n_jobs, total)
        print(f'Processing {total} file(s) with {n_jobs} workers')
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=n_jobs
        ) as executor:
            futures = {
                executor.submit(_process_one_file, fpath, fkey, config): fkey
                for fpath, fkey in zip(all_files, keys)
            }
            done = 0
            for future in concurrent.futures.as_completed(futures):
                fkey, feats, err = future.result()
                done += 1
                print(f'[{done}/{total}] Processed {fkey}')
                if err is not None:
                    errors[fkey] = err
                else:
                    results[fkey] = feats
        # Restore deterministic (sorted) ordering regardless of
        # completion order, so parallel output == serial output.
        results = {k: results[k] for k in keys if k in results}
    else:
        for idx, (fpath, fkey) in enumerate(zip(all_files, keys), 1):
            print(f'[{idx}/{total}] Processing {fkey}')
            fkey, feats, err = _process_one_file(fpath, fkey, config)
            if err is not None:
                errors[fkey] = err
            else:
                results[fkey] = feats

    if errors:
        errors_path = f'{output_path}.errors.json'
        with open(errors_path, 'w') as f:
            json.dump(errors, f, indent=2)
        print(f'WARNING: {len(errors)} file(s) failed; details in {errors_path}')

    if save_format == 'npy':
        np.save(output_path, results)
    elif save_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif save_format in ('csv', 'parquet'):
        rows = {fkey: _flatten_to_columns(feats)
                for fkey, feats in results.items()}
        df = pd.DataFrame.from_dict(rows, orient='index')
        df.index.name = 'file'
        if save_format == 'csv':
            df.to_csv(output_path)
        else:
            try:
                df.to_parquet(output_path)
            except ImportError as e:
                raise ImportError(
                    'Parquet output requires pyarrow. Install it with '
                    '`pip install pyarrow` (or `pip install "AFX[parquet]"`).'
                ) from e
    else:
        raise ValueError(f'Unknown save_format: {save_format}')

    out = dict(results)
    if errors:
        out['errors'] = errors
    return out
