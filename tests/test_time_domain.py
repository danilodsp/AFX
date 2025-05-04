"""
Unit tests for time_domain feature extractors.
"""
import numpy as np
import pytest
from AFX.extractors import time_domain

def test_extract_zero_crossing_rate():
    signal = np.array([0, 1, -1, 1, -1, 0], dtype=np.float32)
    result = time_domain.extract_zero_crossing_rate(signal, sr=22050)
    assert 'zcr' in result
    assert isinstance(result['zcr'], np.ndarray)

def test_extract_variance():
    signal = np.ones(100)
    result = time_domain.extract_variance(signal)
    assert 'variance' in result
    assert np.allclose(result['variance'], 0.0)

def test_extract_entropy():
    signal = np.random.randn(1000)
    result = time_domain.extract_entropy(signal)
    assert 'entropy' in result
    assert result['entropy'].shape == (1,)

def test_extract_crest_factor():
    signal = np.array([0, 1, -1, 1, -1, 0], dtype=np.float32)
    result = time_domain.extract_crest_factor(signal)
    assert 'crest_factor' in result
    assert result['crest_factor'].shape == (1,)

def test_extract_kurtosis():
    signal = np.random.randn(1000)
    from AFX.extractors.time_domain import extract_kurtosis
    result = extract_kurtosis(signal)
    assert 'kurtosis' in result
    assert result['kurtosis'].shape == (1,)

def test_extract_short_time_energy():
    signal = np.ones(100)
    from AFX.extractors.time_domain import extract_short_time_energy
    result = extract_short_time_energy(signal)
    assert 'short_time_energy' in result
    assert result['short_time_energy'].shape == (1,)

def test_extract_energy_ratio():
    signal = np.concatenate([np.ones(50), np.zeros(50)])
    from AFX.extractors.time_domain import extract_energy_ratio
    result = extract_energy_ratio(signal)
    assert 'energy_ratio' in result
    assert result['energy_ratio'].shape == (1,)

def test_extract_sample_entropy():
    signal = np.random.randn(100)
    from AFX.extractors.time_domain import extract_sample_entropy
    result = extract_sample_entropy(signal)
    assert 'sample_entropy' in result
    assert result['sample_entropy'].shape == (1,)

def test_extract_mobility():
    signal = np.random.randn(100)
    from AFX.extractors.time_domain import extract_mobility
    result = extract_mobility(signal)
    assert 'mobility' in result
    assert result['mobility'].shape == (1,)

def test_extract_complexity():
    signal = np.random.randn(100)
    from AFX.extractors.time_domain import extract_complexity
    result = extract_complexity(signal)
    assert 'complexity' in result
    assert result['complexity'].shape == (1,)
