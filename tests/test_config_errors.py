"""
Config validation and provenance serialization: each error names what to fix.
"""
import json

import numpy as np
import pytest

import afx_audio as afx
from afx_audio.provenance import canonical_json


@pytest.mark.parametrize(('config', 'message'), [
    (['zcr'], 'Config root must be a JSON object'),
    ({'features': ['zcr']}, 'Config must specify sample_rate'),
    ({'sample_rate': '16000', 'features': ['zcr']}, 'sample_rate must be a number, got str'),
    ({'sample_rate': True, 'features': ['zcr']}, 'sample_rate must be a number, got bool'),
    ({'sample_rate': 16000, 'features': ['zcr'], 'aggregation_params': [3]},
     'aggregation_params must be a JSON object'),
])
def test_validate_config_errors(config, message):
    with pytest.raises(ValueError, match=message):
        afx.validate_config(config)


def test_sample_rate_optional_when_not_required():
    assert afx.validate_config({'features': ['zcr']}, require_sample_rate=False)['features'] == {'zcr': {}}


def test_load_config_invalid_json(tmp_path):
    path = tmp_path / 'broken.json'
    path.write_text('{"sample_rate": 16000,')
    with pytest.raises(ValueError, match=f'Invalid JSON in config file {path}'):
        afx.load_config(str(path))


def test_load_config_rejects_a_non_object_root(tmp_path):
    path = tmp_path / 'list.json'
    path.write_text(json.dumps(['zcr']))
    with pytest.raises(ValueError, match='Config root must be a JSON object: .*list.json'):
        afx.load_config(str(path))


def test_numpy_values_hash_like_python_values():
    plain = {'sample_rate': 16000, 'features': {'mfcc': {'n_mfcc': 13, 'fmin': 0.0, 'lifter': [1, 2]}}}
    numpy = {'sample_rate': np.int64(16000),
             'features': {'mfcc': {'n_mfcc': np.int32(13), 'fmin': np.float64(0.0), 'lifter': np.array([1, 2])}}}
    assert canonical_json(numpy) == canonical_json(plain)
    assert afx.config_hash(numpy) == afx.config_hash(plain)


def test_unserializable_config_value_names_its_type():
    with pytest.raises(TypeError, match='Config value of type set is not JSON serializable'):
        canonical_json({'features': {'zcr': {'x': {1, 2}}}})
