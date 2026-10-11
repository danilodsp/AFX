"""
Domain feature packs for afx_audio.

Small, curated feature-extraction presets tuned for specific audio domains.
Each preset is a JSON config (same schema as ``afx_audio/config.json``) shipped
inside the package, so it can be loaded without any external files::

    from afx_audio.presets import load_preset, list_presets

    list_presets()
    # ['bioacoustics', 'egemaps', 'machinery', 'music', 'speech']

    config = load_preset('speech')     # latest version
    config = load_preset('speech@1')   # exactly version 1
    feats = extract_all_features(signal, sr, config)

The returned object is a plain ``dict`` validated by
``afx_audio.utils.config_loader.load_config``, so it is a drop-in replacement for a
hand-written config in any AFX entry point (``extract_all_features``,
``extract_features_from_folder``, the sklearn transformer, ...).

Presets are versioned so a paper can cite an exact definition: version N of
preset ``name`` is the file ``name@N.json`` and carries
``"preset": {"name": name, "version": N}``. A released version is never
edited; a change is a new file with the next version number.

Available packs:

- ``speech``       Speech & paralinguistics (emotion, speaker traits).
- ``bioacoustics`` Animal / environmental sound classification.
- ``music``        Music information retrieval (genre, instrument, key).
- ``machinery``    Machinery condition monitoring (fault / anomaly detection).
- ``egemaps``      The 88 eGeMAPS v02 functionals per clip (Eyben et al., 2016).
"""
from __future__ import annotations

import os
import re
from typing import Any

from afx_audio.utils.config_loader import load_config

_PRESET_DIR = os.path.dirname(os.path.abspath(__file__))
_FILE = re.compile(r'([a-z][a-z0-9_]*)@([1-9][0-9]*)\.json')


def _versions() -> dict[str, list[int]]:
    index: dict[str, list[int]] = {}
    for fname in os.listdir(_PRESET_DIR):
        match = _FILE.fullmatch(fname)
        if match:
            index.setdefault(match.group(1), []).append(int(match.group(2)))
    return {name: sorted(versions) for name, versions in sorted(index.items())}


def list_presets(*, versions: bool = False) -> list[str]:
    """Sorted preset names, or every ``name@version`` id when ``versions`` is True."""
    index = _versions()
    if versions:
        return [f'{name}@{v}' for name, vs in index.items() for v in vs]
    return list(index)


def resolve_preset(preset: str) -> str:
    """
    The exact ``name@version`` id for ``'name'`` (latest version) or ``'name@version'``.

    Raises ValueError for an unknown name or version, listing what exists.
    """
    name, sep, version = preset.partition('@')
    index = _versions()
    if name not in index:
        raise ValueError(f'Unknown preset: {preset!r}. Available presets: {list(index)}')
    if not sep:
        return f'{name}@{index[name][-1]}'
    if not version.isdigit() or int(version) not in index[name]:
        available = [f'{name}@{v}' for v in index[name]]
        raise ValueError(f'Unknown version of preset {name!r}: {preset!r}. Available: {available}')
    return f'{name}@{int(version)}'


def load_preset(preset: str) -> dict[str, Any]:
    """
    Load a domain feature pack as a validated AFX config dict.

    Args:
        preset: ``'name'`` for the latest version (e.g. ``'speech'``) or
            ``'name@version'`` for an exact one (e.g. ``'speech@1'``).
    Returns:
        A config dict ready for ``extract_all_features`` /
        ``extract_features_from_folder``. The metadata keys (``preset``,
        ``title``, ``description``) are kept and ignored by the extractors;
        ``preset`` is what provenance records report as ``preset_id``.
    Raises:
        ValueError: If the preset name or version is unknown.
    """
    return load_config(os.path.join(_PRESET_DIR, f'{resolve_preset(preset)}.json'))


def preset_info(preset: str) -> dict[str, Any]:
    """Id, name, version, title, description, sample rate and feature names of a preset."""
    config = load_preset(preset)
    block = config['preset']
    return {
        'id': f"{block['name']}@{block['version']}",
        'name': block['name'],
        'version': block['version'],
        'title': config.get('title', block['name']),
        'description': config.get('description', ''),
        'sample_rate': config['sample_rate'],
        'features': list(config['features']),
    }


def preset_id(config: dict[str, Any]) -> str | None:
    """``'name@version'`` from a config's preset block, None if it has none."""
    block = config.get('preset')
    return None if block is None else f"{block['name']}@{block['version']}"


__all__ = ['list_presets', 'preset_info', 'load_preset', 'resolve_preset', 'preset_id']
