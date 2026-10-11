"""
Tests for the AFX domain feature packs (``afx_audio.presets``).
"""
import json
import os

import numpy as np
import pytest

import afx_audio
from afx_audio import config_hash, extract_all_features
from afx_audio.presets import list_presets, load_preset, preset_id, preset_info, resolve_preset
from afx_audio.utils.config_loader import validate_config

EXPECTED = {'speech', 'bioacoustics', 'music', 'machinery'}

# config_hash of every released preset version. A released version is never
# edited: change a preset by adding name@<next>.json and a line here.
FROZEN = {
    'bioacoustics@1': 'f2fbc8a2bb11',
    'egemaps@1': 'cd530990145e',
    'machinery@1': '563073e5d310',
    'music@1': 'cef990b7ffe1',
    'speech@1': '71f5a1447dc2',
}


def test_list_presets_contains_all_domains():
    names = set(list_presets())
    assert EXPECTED <= names


def test_presets_exposed_at_package_level():
    assert afx_audio.load_preset('speech') == load_preset('speech')
    assert set(afx_audio.list_presets()) >= EXPECTED


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_preset_is_valid_config(name):
    cfg = load_preset(name)
    # load_preset routes through load_config, so these are guaranteed, but
    # assert explicitly to lock the contract.
    assert isinstance(cfg, dict)
    assert isinstance(cfg['sample_rate'], (int, float))
    assert cfg['sample_rate'] > 0
    assert isinstance(cfg['features'], dict) and cfg['features']


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_preset_features_are_known_extractors(name):
    cfg = load_preset(name)
    unknown = set(cfg['features']) - set(afx_audio.FEATURE_MAP)
    assert not unknown, f'{name} references unknown features: {unknown}'


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_preset_info_fields(name):
    info = preset_info(name)
    assert info['name'] == name
    assert info['id'] == f"{name}@{info['version']}" == resolve_preset(name)
    assert info['title'] and info['description']
    assert info['sample_rate'] == load_preset(name)['sample_rate']
    assert info['features'] == list(load_preset(name)['features'])


def test_released_presets_are_frozen():
    """Fails on any edit to a released preset; add a new version instead."""
    assert list_presets(versions=True) == sorted(FROZEN)
    assert {pid: config_hash(load_preset(pid)) for pid in FROZEN} == FROZEN


@pytest.mark.parametrize('pid', sorted(FROZEN))
def test_preset_file_matches_its_id(pid):
    path = os.path.join(os.path.dirname(afx_audio.presets.__file__), f'{pid}.json')
    with open(path, encoding='utf-8') as f:
        raw = json.load(f)
    name, version = pid.split('@')
    assert raw['preset'] == {'name': name, 'version': int(version)}
    assert preset_id(load_preset(pid)) == pid


def test_versioned_ids():
    assert load_preset('speech@1') == load_preset('speech')
    assert resolve_preset('speech') == 'speech@1'
    assert 'speech@1' in list_presets(versions=True)
    assert 'speech' in list_presets()
    with pytest.raises(ValueError, match=r"Available: \['speech@1'\]"):
        load_preset('speech@99')
    with pytest.raises(ValueError, match='Unknown version'):
        load_preset('speech@one')


def test_latest_version_wins(monkeypatch):
    monkeypatch.setattr(afx_audio.presets, '_versions', lambda: {'speech': [1, 2, 10]})
    assert resolve_preset('speech') == 'speech@10'
    assert resolve_preset('speech@2') == 'speech@2'


@pytest.mark.parametrize('block', [
    'speech@1', {'name': 'speech'}, {'name': 'Speech', 'version': 1}, {'name': 'speech', 'version': 0},
    {'name': 'speech', 'version': True}, {'name': 'speech', 'version': 1, 'extra': 1},
])
def test_preset_block_is_validated(block):
    with pytest.raises(ValueError, match='preset'):
        validate_config({'sample_rate': 16000, 'features': ['zcr'], 'preset': block})


def test_unknown_preset_raises():
    with pytest.raises(ValueError):
        load_preset('does-not-exist')


@pytest.mark.parametrize('name', sorted(EXPECTED))
def test_preset_extracts_fixed_length_vector(name):
    """Each preset must turn audio into a finite, fixed-length descriptor."""
    cfg = load_preset(name)
    sr = cfg['sample_rate']
    rng = np.random.default_rng(0)

    def vec_for(seconds):
        n = int(seconds * sr)
        # A voiced component: pitch is NaN on frames without one
        tone = 0.3 * np.sin(2 * np.pi * 150 * np.arange(n) / sr)
        sig = (rng.standard_normal(n) * 0.1 + tone).astype('float32')
        feats = extract_all_features(sig, sr, cfg)
        return np.concatenate([np.asarray(v).ravel() for v in feats.values()])

    v1 = vec_for(1.5)
    v2 = vec_for(2.5)
    assert v1.size == v2.size > 0, 'descriptor length must not depend on clip length'
    assert np.isfinite(v1).all(), f'{name} produced non-finite features'


@pytest.mark.parametrize('pid', sorted(FROZEN))
def test_preset_spells_out_every_parameter(pid):
    # A default that changes in a later release must not change a released preset.
    for name, params in load_preset(pid)['features'].items():
        assert set(params) == set(afx_audio.feature_info(name)['parameters']), name
