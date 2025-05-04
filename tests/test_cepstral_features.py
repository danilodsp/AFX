"""
Unit tests for cepstral_features extractors.
"""
import numpy as np
import pytest
from AFX.extractors import cepstral_features

def test_extract_mfcc():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_mfcc(signal, sr=22050)
    assert 'mfcc' in result
    assert isinstance(result['mfcc'], np.ndarray)

def test_extract_mfcc_delta():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_mfcc_delta(signal, sr=22050)
    assert 'mfcc_delta' in result
    assert isinstance(result['mfcc_delta'], np.ndarray)

def test_extract_chroma_cqt():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_chroma_cqt(signal, sr=22050)
    assert 'chroma_cqt' in result
    assert isinstance(result['chroma_cqt'], np.ndarray)

def test_extract_melspectrogram():
    signal = np.random.randn(22050)
    result = cepstral_features.extract_melspectrogram(signal, sr=22050)
    assert 'melspectrogram' in result
    assert isinstance(result['melspectrogram'], np.ndarray)
