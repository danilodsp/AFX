"""
LPC and formant edge cases: window choice, metadata without sr, fewer resonances than requested.
"""
import numpy as np
import pytest
from scipy.signal import get_window, lfilter

from afx_audio.extractors.linear_prediction import _symmetric_window, extract_formants, extract_lpc

SR = 8000


@pytest.mark.parametrize('name', ['hamming', 'hann', 'blackman'])
def test_windows_are_symmetric(name):
    window = _symmetric_window(name, 64)
    np.testing.assert_allclose(window, window[::-1], atol=1e-15)
    np.testing.assert_allclose(window, get_window(name, 64, fftbins=False), atol=1e-15)


def test_window_changes_the_fit():
    x = np.random.default_rng(0).standard_normal(1024)
    hann = extract_lpc(x, SR, order=4, window='hann')['lpc']
    rect = extract_lpc(x, SR, order=4, window=None)['lpc']
    assert hann.shape == rect.shape == (4, 1)
    assert not np.allclose(hann, rect)


def test_lpc_metadata_needs_sr():
    with pytest.raises(ValueError, match='sr is required when return_metadata=True'):
        extract_lpc(np.ones(256), None, frame_size=64, hop_length=32, return_metadata=True)


def test_order_two_finds_one_formant_and_pads_the_rest_with_nan():
    # Two-pole resonator at 1 kHz driven by noise: one spectral peak
    r, f0 = 0.97, 1000.0
    a = [1.0, -2 * r * np.cos(2 * np.pi * f0 / SR), r * r]
    x = lfilter([1.0], a, np.random.default_rng(1).standard_normal(SR))
    formants = extract_formants(x, SR, n_formants=3, order=2, pre_emphasis=0.0, frame_size=512, hop_length=256)
    formants = formants['formants']
    assert formants.shape[0] == 3
    assert np.nanmedian(formants[0]) == pytest.approx(f0, rel=0.05)
    assert np.isnan(formants[1:]).all()
