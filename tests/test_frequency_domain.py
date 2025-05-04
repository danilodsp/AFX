"""
Unit tests for frequency_domain feature extractors.
"""
import numpy as np
import pytest
from AFX.extractors import frequency_domain

def test_extract_spectral_centroid():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_centroid(signal, sr=22050)
    assert 'spectral_centroid' in result
    assert isinstance(result['spectral_centroid'], np.ndarray)

def test_extract_spectral_bandwidth():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_bandwidth(signal, sr=22050)
    assert 'spectral_bandwidth' in result
    assert isinstance(result['spectral_bandwidth'], np.ndarray)

def test_extract_spectral_rolloff():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_rolloff(signal, sr=22050)
    assert 'spectral_rolloff' in result
    assert isinstance(result['spectral_rolloff'], np.ndarray)
