"""
Per-column dataset statistics over flattened feature vectors, in one pass.

Means and variances use Welford's update for single vectors and the
pairwise merge of Chan, Golub and LeVeque (1979) for combining shards, so
statistics of a dataset processed in chunks equal those of a single pass.
NaN values are counted and skipped, as in ``numpy.nanmean`` and
``numpy.nanvar`` (population variance, ddof=0).
"""
import json
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

STATS_SUFFIX = '.stats.json'


class DatasetStats:
    """Streaming count, mean, variance, min, max and NaN count per named column."""

    def __init__(self) -> None:
        self.columns: list[str] = []
        self._index: dict[str, int] = {}
        self.n_rows = 0
        self.count = np.zeros(0, dtype=np.int64)
        self.nan_count = np.zeros(0, dtype=np.int64)
        self.mean = np.zeros(0)
        self.m2 = np.zeros(0)
        self.min = np.zeros(0)
        self.max = np.zeros(0)

    def _grow(self, names: Sequence[str]) -> np.ndarray:
        """Column indices for names, appending columns not seen before."""
        new = [n for n in dict.fromkeys(names) if n not in self._index]
        if new:
            k = len(new)
            for n in new:
                self._index[n] = len(self.columns)
                self.columns.append(n)
            self.count = np.concatenate([self.count, np.zeros(k, dtype=np.int64)])
            self.nan_count = np.concatenate([self.nan_count, np.zeros(k, dtype=np.int64)])
            self.mean = np.concatenate([self.mean, np.zeros(k)])
            self.m2 = np.concatenate([self.m2, np.zeros(k)])
            self.min = np.concatenate([self.min, np.full(k, np.inf)])
            self.max = np.concatenate([self.max, np.full(k, -np.inf)])
        return np.array([self._index[n] for n in names], dtype=np.intp)

    def update(self, names: Sequence[str], values: np.ndarray) -> None:
        """Add one row (Welford's update on the non-NaN entries)."""
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (len(names),):
            raise ValueError(f'values must be 1-D with one entry per name, got shape {values.shape} '
                             f'for {len(names)} names')
        idx = self._grow(names)
        self.n_rows += 1
        nan = np.isnan(values)
        self.nan_count[idx[nan]] += 1
        idx, x = idx[~nan], values[~nan]
        self.count[idx] += 1
        delta = x - self.mean[idx]
        self.mean[idx] += delta / self.count[idx]
        self.m2[idx] += delta * (x - self.mean[idx])
        self.min[idx] = np.minimum(self.min[idx], x)
        self.max[idx] = np.maximum(self.max[idx], x)

    def merge(self, other: 'DatasetStats') -> 'DatasetStats':
        """Fold another shard into this one (pairwise update of Chan, Golub and LeVeque 1979)."""
        idx = self._grow(other.columns)
        self.n_rows += other.n_rows
        self.nan_count[idx] += other.nan_count
        n_a, n_b = self.count[idx].astype(np.float64), other.count.astype(np.float64)
        n = n_a + n_b
        safe_n = np.where(n > 0, n, 1.0)
        delta = other.mean - self.mean[idx]
        self.mean[idx] = np.where(n > 0, self.mean[idx] + delta * n_b / safe_n, 0.0)
        self.m2[idx] += other.m2 + delta**2 * n_a * n_b / safe_n
        self.count[idx] += other.count
        self.min[idx] = np.minimum(self.min[idx], other.min)
        self.max[idx] = np.maximum(self.max[idx], other.max)
        return self

    def to_dict(self) -> dict[str, Any]:
        """Plain arrays; columns with no non-NaN value get NaN mean, var, min and max."""
        empty = self.count == 0
        with np.errstate(invalid='ignore', divide='ignore'):
            var = self.m2 / self.count
        return {
            'n_rows': self.n_rows,
            'columns': list(self.columns),
            'count': self.count.copy(),
            'nan_count': self.nan_count.copy(),
            'mean': np.where(empty, np.nan, self.mean),
            'var': np.where(empty, np.nan, var),
            'min': np.where(empty, np.nan, self.min),
            'max': np.where(empty, np.nan, self.max),
        }


_ARRAY_KEYS = ('count', 'nan_count', 'mean', 'var', 'min', 'max')


def _null_nan(arr: np.ndarray) -> list:
    return [None if not np.isfinite(v) else v for v in arr.tolist()]


def write_stats(path: str, stats: dict[str, Any], **extra: Any) -> None:
    """Write to_dict() output as JSON (NaN and infinity as null)."""
    record = {**extra, 'n_rows': stats['n_rows'], 'columns': stats['columns']}
    for key in _ARRAY_KEYS:
        values = stats[key]
        record[key] = values.tolist() if key.endswith('count') else _null_nan(values)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, allow_nan=False)


def load_stats(path: str | os.PathLike) -> dict[str, Any]:
    """
    Dataset statistics written by ``extract_features_from_folder(..., dataset_stats=True)``.

    ``path`` is the batch output or its ``.stats.json`` file. Returns the
    record with ``count`` and ``nan_count`` as int64 arrays and ``mean``,
    ``var``, ``std``, ``min``, ``max`` as float64 arrays (NaN where a column
    had no non-NaN value), aligned with ``columns``.
    """
    path = os.fspath(path)
    if not path.endswith(STATS_SUFFIX):
        path = f'{path}{STATS_SUFFIX}'
    if not os.path.isfile(path):
        raise FileNotFoundError(f'No dataset statistics at {path}; run the batch with dataset_stats=True (afx --stats)')
    with open(path, encoding='utf-8') as f:
        record = json.load(f)
    for key in _ARRAY_KEYS:
        dtype = np.int64 if key.endswith('count') else np.float64
        record[key] = np.array([np.nan if v is None else v for v in record[key]], dtype=dtype)
    record['std'] = np.sqrt(record['var'])
    return record
