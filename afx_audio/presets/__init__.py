"""
Domain feature packs for afx_audio.

Small, curated feature-extraction presets tuned for specific audio domains.
Each preset is a JSON config (same schema as ``afx_audio/config.json``) shipped
inside the package, so it can be loaded without any external files::

    from afx_audio.presets import load_preset, list_presets

    list_presets()
    # ['bioacoustics', 'machinery', 'music', 'speech']

    config = load_preset('speech')
    feats = extract_all_features(signal, sr, config)

The returned object is a plain ``dict`` validated by
``afx_audio.utils.config_loader.load_config``, so it is a drop-in replacement for a
hand-written config in any AFX entry point (``extract_all_features``,
``extract_features_from_folder``, the sklearn transformer, ...).

Available packs:

- ``speech``       Speech & paralinguistics (emotion, speaker traits).
- ``bioacoustics`` Animal / environmental sound classification.
- ``music``        Music information retrieval (genre, instrument, key).
- ``machinery``    Machinery condition monitoring (fault / anomaly detection).
"""
from __future__ import annotations

import json
import os
from typing import Any

from afx_audio.utils.config_loader import load_config

_PRESET_DIR = os.path.dirname(os.path.abspath(__file__))


def _preset_path(name: str) -> str:
    return os.path.join(_PRESET_DIR, f'{name}.json')


def list_presets() -> list[str]:
    """Return the sorted names of all available domain feature packs."""
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(_PRESET_DIR)
        if f.endswith('.json')
    )


def preset_info(name: str) -> dict[str, str]:
    """
    Return the human-readable metadata for a preset without loading it as a
    full extraction config.

    Args:
        name: Preset name (see :func:`list_presets`).
    Returns:
        Dict with ``name``, ``title`` and ``description`` keys.
    """
    raw = _load_raw(name)
    return {
        'name': raw.get('name', name),
        'title': raw.get('title', name),
        'description': raw.get('description', ''),
    }


def _load_raw(name: str) -> dict[str, Any]:
    path = _preset_path(name)
    if not os.path.exists(path):
        raise ValueError(
            f'Unknown preset: {name!r}. Available presets: {list_presets()}'
        )
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_preset(name: str) -> dict[str, Any]:
    """
    Load a domain feature pack as a validated AFX config dict.

    Args:
        name: Preset name, one of :func:`list_presets` (e.g. ``'speech'``).
    Returns:
        A config dict ready for ``extract_all_features`` /
        ``extract_features_from_folder``. The extra metadata keys
        (``name``, ``title``, ``description``) are preserved and ignored by
        the extractors.
    Raises:
        ValueError: If the preset name is unknown or the JSON is invalid.
    """
    path = _preset_path(name)
    if not os.path.exists(path):
        raise ValueError(
            f'Unknown preset: {name!r}. Available presets: {list_presets()}'
        )
    # Reuse the standard loader so presets are validated exactly like any
    # user config (sample_rate + features block required).
    return load_config(path)


__all__ = ['list_presets', 'preset_info', 'load_preset']
