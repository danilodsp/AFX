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

def test_extract_spectral_flux():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_flux(signal, sr=22050)
    assert 'spectral_flux' in result
    assert isinstance(result['spectral_flux'], np.ndarray)

def test_extract_spectral_skewness():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_skewness(signal, sr=22050)
    assert 'spectral_skewness' in result
    assert isinstance(result['spectral_skewness'], np.ndarray)

def test_extract_spectral_slope():
    signal = np.random.randn(22050)
    result = frequency_domain.extract_spectral_slope(signal, sr=22050)
    assert 'spectral_slope' in result
    assert isinstance(result['spectral_slope'], np.ndarray)
