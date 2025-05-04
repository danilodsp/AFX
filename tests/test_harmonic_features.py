"""
Unit tests for harmonic_features extractors.
"""
import numpy as np
import pytest
from AFX.extractors import harmonic_features

def test_extract_pitch():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_pitch(signal, sr=22050)
    assert 'pitch' in result
    assert isinstance(result['pitch'], np.ndarray)

def test_extract_thd():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_thd(signal, sr=22050)
    assert 'thd' in result
    assert result['thd'].shape == (1,)

def test_extract_hnr():
    signal = np.random.randn(22050)
    result = harmonic_features.extract_hnr(signal, sr=22050)
    assert 'hnr' in result
    assert isinstance(result['hnr'], np.ndarray)
