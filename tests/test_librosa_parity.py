"""Features documented as equal to a librosa function, checked to numerical precision."""
import numpy as np
import pytest

librosa = pytest.importorskip('librosa')

import afx_audio  # noqa: E402

F = afx_audio.FEATURE_MAP


def _signal(sr, seconds=1.5, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    tones = 0.6 * np.sin(2 * np.pi * 220 * t) + 0.3 * np.sin(2 * np.pi * 659 * t)
    return tones * (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * t)) + 0.05 * rng.standard_normal(len(t))


FRAMINGS = [(22050, 2048, 512, False), (22050, 2048, 512, True), (16000, 400, 160, False), (16000, 512, 128, True)]


def _close(ours, ref, rtol=1e-6, atol=1e-6):
    ref = np.asarray(ref)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=rtol, atol=atol * max(1.0, np.abs(ref).max()))


@pytest.mark.parametrize('sr, n_fft, hop, center', FRAMINGS)
class TestSpectral:
    def test_centroid(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['spectral_centroid'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['spectral_centroid']
        _close(ours, librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=center)[0])

    def test_bandwidth(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['spectral_bandwidth'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['spectral_bandwidth']
        _close(ours, librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=center)[0])

    def test_rolloff(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['spectral_rolloff'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['spectral_rolloff']
        _close(ours, librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=center)[0])

    def test_flatness(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['spectral_flatness'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['spectral_flatness']
        _close(ours, librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop, center=center)[0])

    def test_contrast(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['spectral_contrast'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['spectral_contrast']
        ref = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=center)
        _close(ours, ref, rtol=1e-5, atol=1e-5)

    def test_melspectrogram(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['melspectrogram'](y, sr, n_mels=64, frame_size=n_fft, hop_length=hop, center=center)['melspectrogram']
        _close(ours, librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=64,
                                                    center=center, pad_mode='constant'), rtol=1e-5)

    def test_mfcc(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['mfcc'](y, sr, n_mfcc=20, frame_size=n_fft, hop_length=hop, n_mels=40, center=center)['mfcc']
        ref = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop, n_mels=40, center=center,
                                   pad_mode='constant')
        _close(ours, ref, rtol=1e-5, atol=1e-5)

    def test_mfcc_deltas(self, sr, n_fft, hop, center):
        y = _signal(sr)
        kw = dict(n_mfcc=13, frame_size=n_fft, hop_length=hop, center=center)
        mfcc = F['mfcc'](y, sr, **kw)['mfcc']
        d1 = F['mfcc_delta'](y, sr, **kw)['mfcc_delta']
        d2 = F['mfcc_delta_delta'](y, sr, **kw)['mfcc_delta_delta']
        _close(d1, librosa.feature.delta(mfcc), atol=1e-9)
        _close(d2, librosa.feature.delta(mfcc, order=2), atol=1e-9)

    def test_chroma_stft(self, sr, n_fft, hop, center):
        y = _signal(sr)
        ours = F['chroma_stft'](y, sr, frame_size=n_fft, hop_length=hop, center=center)['chroma_stft']
        ref = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop, center=center, tuning=0.0)
        _close(ours, ref, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize('n_fft, hop', [(2048, 512), (1024, 256), (512, 128)])
def test_tonnetz_and_onset_strength(n_fft, hop):
    sr = 22050
    y = _signal(sr)
    y[::5000] += 2.0  # clicks give the onset envelope something to find
    ours = F['tonnetz'](y, sr, frame_size=n_fft, hop_length=hop)['tonnetz']
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop, tuning=0.0)
    _close(ours, librosa.feature.tonnetz(chroma=chroma), atol=1e-6)
    for lag, max_size in [(1, 1), (2, 3)]:
        ours = F['onset_strength'](y, sr, frame_size=n_fft, hop_length=hop, lag=lag, max_size=max_size)
        ref = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft, hop_length=hop, lag=lag, max_size=max_size)
        _close(ours['onset_strength'], ref, rtol=1e-6)


@pytest.mark.parametrize('center', [False, True])
@pytest.mark.parametrize('frame, hop', [(2048, 512), (400, 160), (512, 512), (401, 100)])
def test_zcr_and_rms_match_librosa(frame, hop, center):
    y = _signal(16000)
    y[:2000] = 1e-12  # near-silence must not count as crossings
    zcr = F['zcr'](y, 16000, frame_size=frame, hop_length=hop, center=center)['zcr']
    ref = librosa.feature.zero_crossing_rate(y, frame_length=frame, hop_length=hop, center=center)[0]
    _close(zcr, ref, atol=1e-12)
    rms = F['rms_energy'](y, 16000, frame_size=frame, hop_length=hop, center=center)['rms_energy']
    _close(rms, librosa.feature.rms(y=y, frame_length=frame, hop_length=hop, center=center)[0], rtol=1e-6)


def test_yin_close_to_librosa():
    sr = 22050
    t = np.arange(2 * sr) / sr
    f0 = 120 + 80 * t  # slow glide
    y = np.sin(2 * np.pi * np.cumsum(f0) / sr) + 0.01 * np.random.default_rng(1).standard_normal(len(t))
    ours = F['pitch'](y, sr, frame_size=2048, hop_length=512, fmin=60, fmax=800)['pitch']
    ref = librosa.yin(y, fmin=60, fmax=800, sr=sr, frame_length=2048, hop_length=512, center=False)
    assert ours.shape == ref.shape
    np.testing.assert_allclose(ours, ref, rtol=0.02)  # librosa 0.10 and 1.0 differ by about 1%
