"""
Provenance records for batch outputs: what produced a feature file.

Parquet outputs carry the record in the schema metadata under the key
``afx``; every other format gets a ``<output>.provenance.json`` sidecar.
"""
import datetime
import hashlib
import json
import os
import platform
import sys
from typing import Any

import numpy as np

PROVENANCE_KEY = 'afx'
SIDECAR_SUFFIX = '.provenance.json'


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Config value of type {type(value).__name__} is not JSON serializable')


def canonical_json(obj: Any) -> str:
    """Sorted keys, no whitespace, NaN rejected: the same dict always gives the same string."""
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False, default=_json_default)


def config_hash(config: dict[str, Any]) -> str:
    """
    First 12 hex characters of the sha256 of the config's canonical JSON.

    Key order does not change the hash. Batch outputs hash the validated
    config they record, so ``config_hash(read_provenance(p)['config'])``
    equals ``read_provenance(p)['config_hash']``.
    """
    if not isinstance(config, dict):
        raise TypeError(f'config must be a dict, got {type(config).__name__}')
    return hashlib.sha256(canonical_json(config).encode('utf-8')).hexdigest()[:12]


def file_sha256(path: str, chunk_size: int = 1 << 20) -> str:
    """Hex sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _versions() -> dict[str, str]:
    import scipy
    import soundfile

    from afx_audio import __version__

    return {
        'afx_audio': __version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'soundfile': soundfile.__version__,
        'libsndfile': soundfile.__libsndfile_version__,
        'python': platform.python_version(),
    }


def environment_record() -> dict[str, Any]:
    """Library versions, platform and the UTC creation time."""
    return {
        'versions': _versions(),
        'platform': sys.platform,
        'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
    }


def build_provenance(
    config: dict[str, Any],
    *,
    output_format: str,
    native_rates: dict[str, int],
    file_hashes: dict[str, str] | None,
    sample_rate_overridden: bool,
    n_files: int,
    n_errors: int,
    custom_features: list[str],
) -> dict[str, Any]:
    """The record written next to a batch output."""
    target = config['sample_rate']
    rate_counts: dict[str, int] = {}
    for rate in native_rates.values():
        rate_counts[str(rate)] = rate_counts.get(str(rate), 0) + 1
    return {
        'afx_version': _versions()['afx_audio'],
        'config_hash': config_hash(config),
        'config': json.loads(canonical_json(config)),
        'sample_rate': target,
        'resampling': {
            'method': 'scipy.signal.resample_poly',
            'sample_rate_overridden': sample_rate_overridden,
            'native_sample_rates': dict(sorted(rate_counts.items(), key=lambda kv: int(kv[0]))),
            'n_resampled': sum(rate != target for rate in native_rates.values()),
        },
        'format': output_format,
        'n_files': n_files,
        'n_errors': n_errors,
        'custom_features': custom_features,
        'file_sha256': file_hashes,
        **environment_record(),
    }


def sidecar_path(output_path: str) -> str:
    return f'{output_path}{SIDECAR_SUFFIX}'


def write_sidecar(output_path: str, record: dict[str, Any]) -> str:
    path = sidecar_path(output_path)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(record, f, indent=2, allow_nan=False)
    return path


def read_provenance(path: str | os.PathLike) -> dict[str, Any]:
    """
    Provenance record of a batch output.

    ``path`` is the output itself (parquet metadata or the sidecar next to
    it is read) or the ``.provenance.json`` sidecar.
    """
    path = os.fspath(path)
    if path.endswith(SIDECAR_SUFFIX):
        sidecar = path
    elif path.lower().endswith('.parquet'):
        import pyarrow.parquet as pq

        metadata = pq.read_schema(path).metadata or {}
        raw = metadata.get(PROVENANCE_KEY.encode())
        if raw is None:
            raise ValueError(f'{path} has no {PROVENANCE_KEY!r} provenance metadata; '
                             'it was not written by an afx-audio version that records provenance')
        return json.loads(raw)
    else:
        sidecar = sidecar_path(path)
    if not os.path.isfile(sidecar):
        raise FileNotFoundError(f'No provenance record for {path}: expected {sidecar}')
    with open(sidecar, encoding='utf-8') as f:
        return json.load(f)
