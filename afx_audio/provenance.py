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

from afx_audio.presets import list_presets, load_preset, preset_id

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


# Values a config takes when it leaves a key out; filled in before hashing.
_COMPUTE_DEFAULTS: dict[str, Any] = {
    'aggregation': 'mean',
    'aggregation_params': {},
    'normalize': None,
    'preserve_shape': False,
    'coverage': False,
}
# Labels and output layout, not what is computed.
_NOT_HASHED = ('title', 'description', 'preset', 'flatten')


def computed_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    The part of a config that decides the feature values, with every default written out.

    Feature parameters are merged over the extractor's defaults and the
    top-level options over theirs; labels (title, description, the preset
    block) and ``flatten`` are dropped. Two configs that compute the same
    values give the same dict, whichever front end filled them in.
    """
    from afx_audio.registry import feature_parameters

    if not isinstance(config, dict):
        raise TypeError(f'config must be a dict, got {type(config).__name__}')
    out = {k: v for k, v in config.items() if k not in _NOT_HASHED}
    for key, default in _COMPUTE_DEFAULTS.items():
        if out.get(key) is None:
            out[key] = default
    features = config.get('features', {})
    if isinstance(features, list):
        features = {name: {} for name in features}
    out['features'] = {name: {**feature_parameters(name), **(params or {})} for name, params in features.items()}
    return out


def config_hash(config: dict[str, Any]) -> str:
    """
    First 12 hex characters of the sha256 of ``computed_config(config)`` as canonical JSON.

    Key order, labels, ``flatten`` and parameters written at their default
    value do not change the hash, so a preset, the same preset copied into
    a file, and the config the scikit-learn transformer builds from it all
    hash alike.
    """
    return hashlib.sha256(canonical_json(computed_config(config)).encode('utf-8')).hexdigest()[:12]


def preset_record(config: dict[str, Any]) -> dict[str, Any]:
    """
    ``preset_id`` ('speech@1', None without a preset block) and ``preset_modified``.

    ``preset_modified`` is True when a key the bundled preset sets
    (sample_rate, features and their parameters, aggregation, ...) has a
    different value in ``config``, None when the id is not a bundled
    preset. Options the preset leaves unset (normalize, coverage, flatten)
    are recorded in the config but do not count as a modification.
    """
    pid = preset_id(config)
    modified = None
    if pid is not None and pid in list_presets(versions=True):
        preset = load_preset(pid)
        modified = any(canonical_json(config.get(k)) != canonical_json(v) for k, v in preset.items())
    return {'preset_id': pid, 'preset_modified': modified}


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
        **preset_record(config),
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
