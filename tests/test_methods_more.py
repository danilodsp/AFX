"""
Coverage for afx_audio.methods modules previously untested directly:
stft (incl. result cache), delta, gammatone, cqt, chroma, and the
filterbank caches.
"""
import librosa
import numpy as np
import pytest

from afx_audio.methods.chroma import chroma_from_stft, frequency_to_chroma_bin
from afx_audio.methods.cqt import cqt_approximation, cqt_filterbank
from afx_audio.methods.delta import compute_delta
from afx_audio.methods.gammatone import erb_space, gammatone_filterbank
from afx_audio.methods.mel import mel_filterbank
from afx_audio.methods.stft import clear_stft_cache, stft
from afx_audio.utils.cache import cache_mel_filterbank, clear_feature_cache, get_cached_mel_filterbank

SR = 22050


def _tone(freq=440.0, duration=1.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


# ---------- stft ----------

def test_stft_matches_librosa():
    y = _tone(440.0)
    # Both default to center=True with constant padding (librosa >= 0.10)
    ours = stft(y, frame_size=2048, hop_length=512, center=True)
    ref = librosa.stft(y, n_fft=2048, hop_length=512, center=True)
    n = min(ours.shape[1], ref.shape[1])
    assert np.allclose(np.abs(ours[:, :n]), np.abs(ref[:, :n]),
                       rtol=1e-4, atol=1e-6)


def test_stft_result_cache_shares_computation():
    clear_stft_cache()
    y = _tone(440.0, duration=0.5)
    a = stft(y, frame_size=1024, hop_length=256)
    b = stft(y, frame_size=1024, hop_length=256)
    assert a is b  # second call served from cache
    assert not a.flags.writeable
    # Different params -> different result
    c = stft(y, frame_size=1024, hop_length=512)
    assert c is not a
    clear_stft_cache()
    d = stft(y, frame_size=1024, hop_length=256)
    assert d is not a
    assert np.allclose(d, a)


def test_stft_float32_gives_complex64():
    y = _tone(440.0, duration=0.2).astype(np.float32)
    assert stft(y, frame_size=512, hop_length=128).dtype == np.complex64


def test_stft_rejects_empty_and_2d():
    with pytest.raises(ValueError):
        stft(np.zeros(0))
    with pytest.raises(ValueError):
        stft(np.zeros((10, 2)))


def test_stft_short_signal_pads_to_one_frame():
    y = _tone(440.0, duration=0.01)  # 220 samples
    out = stft(y, frame_size=2048, hop_length=512, center=False)
    assert out.shape == (1025, 1)


# ---------- delta ----------

@pytest.mark.parametrize('order', [1, 2])
@pytest.mark.parametrize('width', [3, 5, 9])
def test_compute_delta_matches_librosa(width, order):
    rng = np.random.RandomState(0)
    feats = rng.randn(5, 60)
    ours = compute_delta(feats, width=width, order=order)
    ref = librosa.feature.delta(feats, width=width, order=order, axis=1)
    np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_compute_delta_validates_order():
    with pytest.raises(ValueError, match='order'):
        compute_delta(np.zeros((3, 20)), order=3)


def test_compute_delta_clips_width_to_sequence():
    feats = np.random.RandomState(1).randn(2, 6)
    np.testing.assert_allclose(compute_delta(feats, width=9), compute_delta(feats, width=5))


# ---------- gammatone ----------

@pytest.mark.parametrize('n_fft', [2048, 551])
def test_gammatone_filters_peak_at_center_frequencies(n_fft):
    # 551 is 25 ms at 22050 Hz: for odd n_fft the bins are k sr / n_fft and
    # the top one is below sr / 2, so a linspace(0, sr / 2) axis drifts by up
    # to sr / (2 n_fft) and the peaks land a bin off
    fb = gammatone_filterbank(24, n_fft, SR, 50.0, None)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / SR)
    cf = erb_space(50.0, SR / 2.0, 24)
    nearest = np.argmin(np.abs(freqs[np.newaxis, :] - cf[:, np.newaxis]), axis=1)
    np.testing.assert_array_equal(np.argmax(fb, axis=1), nearest)


def test_gammatone_unit_area_and_cached():
    fb1 = gammatone_filterbank(16, 1024, SR, 50.0, None)
    assert np.allclose(fb1.sum(axis=1), 1.0)
    fb2 = gammatone_filterbank(16, 1024, SR, 50.0, None)
    assert fb1 is fb2  # lru_cache
    assert not fb1.flags.writeable


# ---------- cqt ----------

def test_cqt_filterbank_rejects_super_nyquist():
    with pytest.raises(ValueError, match='Nyquist'):
        cqt_filterbank(n_bins=120, bins_per_octave=12,
                       fmin=32.703, sr=8000, frame_size=2048)


def test_cqt_rows_normalized_and_tone_lands_on_right_bin():
    fb, freqs = cqt_filterbank(84, 12, 32.70319566257483, SR, 2048)
    assert np.allclose(fb.sum(axis=1), 1.0)
    # A4 = 440 Hz is CQT bin 45 when fmin = C1
    cqt_mag, cqt_freqs = cqt_approximation(_tone(440.0), SR, frame_size=2048)
    peak_bin = int(np.argmax(cqt_mag.mean(axis=1)))
    expected_bin = int(np.argmin(np.abs(cqt_freqs - 440.0)))
    assert abs(peak_bin - expected_bin) <= 1


# ---------- chroma ----------

def test_chroma_from_stft_tone_maps_to_pitch_class():
    y = _tone(261.626)  # C4
    mag = np.abs(stft(y, frame_size=4096, hop_length=512))
    chroma = chroma_from_stft(mag, SR, 4096)
    assert chroma.shape[0] == 12
    assert int(np.argmax(chroma.mean(axis=1))) == 0  # C


def test_frequency_to_chroma_bin_validates():
    assert frequency_to_chroma_bin(440.0) == 9  # A
    with pytest.raises(ValueError):
        frequency_to_chroma_bin(0.0)


# ---------- mel filterbank cache ----------

def test_mel_filterbank_cache_roundtrip_and_readonly():
    clear_feature_cache()
    assert get_cached_mel_filterbank(40, 1024, SR, 0.0, SR / 2) is None
    fb = mel_filterbank(40, 1024, SR, 0.0, SR / 2, norm='slaney')
    cache_mel_filterbank(fb, 40, 1024, SR, 0.0, SR / 2)
    cached = get_cached_mel_filterbank(40, 1024, SR, 0.0, SR / 2)
    assert cached is fb
    assert not cached.flags.writeable  # cache is corruption-proof
    # int/float key normalization: 0 == 0.0
    assert get_cached_mel_filterbank(40, 1024, SR, 0, SR // 2 * 2 / 2) is fb
    clear_feature_cache()


def test_mel_filterbank_matches_librosa():
    fb = mel_filterbank(128, 2048, SR, 0.0, SR / 2.0, norm='slaney')
    ref = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128)
    assert np.max(np.abs(fb - ref)) < 1e-6


def test_stft_cache_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    rng = np.random.default_rng(0)
    signals = [rng.standard_normal(4000) for _ in range(12)]
    expected = [np.array(stft(x, frame_size=256, hop_length=64)) for x in signals]
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(lambda i: stft(signals[i % 12], frame_size=256, hop_length=64), range(400)))
    for i, r in enumerate(results):
        np.testing.assert_array_equal(r, expected[i % 12])


def test_delta_kernel_values_and_checks():
    from afx_audio.methods.delta import delta_kernel

    np.testing.assert_allclose(delta_kernel(5), [-0.2, -0.1, 0.0, 0.1, 0.2])
    for bad in (1, 4):
        with pytest.raises(ValueError):
            delta_kernel(bad)


def test_frame_signal_matches_stft_framing():
    from afx_audio.utils.frame_utils import frame_signal

    x = np.arange(10.0)
    frames = frame_signal(x, frame_length=4, hop_length=3)
    np.testing.assert_array_equal(frames, [[0, 3, 6], [1, 4, 7], [2, 5, 8], [3, 6, 9]])
    np.testing.assert_array_equal(frame_signal(np.ones(2), 4, 2)[:, 0], [1, 1, 0, 0])
