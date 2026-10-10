"""
Tests for the AFX domain feature packs (``afx_audio.presets``).
"""
import numpy as np
import pytest

import afx_audio
from afx_audio import extract_all_features
from afx_audio.presets import list_presets, load_preset, preset_info

EXPECTED = {'speech', 'bioacoustics', 'music', 'machinery'}


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
    assert info['title'] and info['description']


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
        sig = (rng.standard_normal(int(seconds * sr)) * 0.1).astype('float32')
        feats = extract_all_features(sig, sr, cfg)
        return np.concatenate([np.asarray(v).ravel() for v in feats.values()])

    v1 = vec_for(1.5)
    v2 = vec_for(2.5)
    assert v1.size == v2.size > 0, 'descriptor length must not depend on clip length'
    assert np.isfinite(v1).all(), f'{name} produced non-finite features'
