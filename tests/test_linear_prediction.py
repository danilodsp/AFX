import numpy as np
import pytest
import scipy.linalg
import scipy.signal

import afx_audio
from afx_audio.extractors.linear_prediction import extract_formants, extract_lpc
from afx_audio.methods.autocorrelation import autocorrelation
from afx_audio.methods.lpc import lpc, lpc_roots

SR = 16000


def _vowel(formants=((700, 60), (1220, 70), (2600, 110)), f0=120.0, seconds=1.0):
    n = int(SR * seconds)
    x = np.zeros(n)
    x[::int(SR / f0)] = 1.0
    for f, bw in formants:
        r = np.exp(-np.pi * bw / SR)
        x = scipy.signal.lfilter([1.0], [1.0, -2 * r * np.cos(2 * np.pi * f / SR), r * r], x)
    return x + 1e-4 * np.random.default_rng(0).standard_normal(n)


def test_lpc_solves_the_normal_equations():
    frames = np.random.default_rng(1).standard_normal((6, 300))
    a, err = lpc(frames, 10)
    for frame, coef, e in zip(frames, a, err, strict=True):
        r = autocorrelation(frame, 11)
        np.testing.assert_allclose(coef[1:], scipy.linalg.solve_toeplitz(r[:10], -r[1:11]), rtol=1e-8)
        assert e == pytest.approx(r[0] + coef[1:] @ r[1:11], rel=1e-8)


def test_lpc_roots_match_numpy_roots():
    a, _ = lpc(np.random.default_rng(2).standard_normal((3, 200)), 8)
    for row, roots in zip(a, lpc_roots(a), strict=True):
        np.testing.assert_allclose(np.sort_complex(roots), np.sort_complex(np.roots(row)), atol=1e-10)


def test_lpc_recovers_an_ar_process():
    rng = np.random.default_rng(3)
    true = np.array([1.0, -1.6, 0.9])
    x = scipy.signal.lfilter([1.0], true, rng.standard_normal(20000))
    coef = extract_lpc(x, SR, order=2, window=None)['lpc'][:, 0]
    np.testing.assert_allclose(coef, true[1:], atol=0.02)


def test_formants_of_a_synthetic_vowel():
    formants = extract_formants(_vowel(), SR, frame_size=512, hop_length=256)['formants']
    np.testing.assert_allclose(np.nanmedian(formants, axis=1), [700, 1220, 2600], rtol=0.05)


def test_formants_nan_on_silence_and_short_frames():
    assert np.isnan(extract_formants(np.zeros(4000), SR)['formants']).all()
    assert np.isnan(extract_formants(np.ones(5), SR)['formants']).all()
    assert np.isnan(extract_lpc(np.ones(5), SR, order=12)['lpc']).all()


def test_shapes_and_times():
    x = _vowel()
    out = afx_audio.FEATURE_MAP['lpc'](x, SR, order=10, frame_size=400, hop_length=160, return_metadata=True)
    assert out['lpc'].shape == (10, 1 + (SR - 400) // 160)
    assert out['metadata']['times'][0] == pytest.approx(200 / SR)
    out = afx_audio.FEATURE_MAP['formants'](x, SR, n_formants=4, return_metadata=True)
    assert out['formants'].shape[0] == 4 and len(out['metadata']['times']) == out['formants'].shape[1]
