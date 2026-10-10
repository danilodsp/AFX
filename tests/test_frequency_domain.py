"""
Unit tests for frequency_domain feature extractors.
"""
import numpy as np

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


def _mag(signal, n_fft=1024, hop=256):
    from AFX.methods.stft import stft
    return np.abs(stft(signal, frame_size=n_fft, hop_length=hop, center=False))


def test_spectral_kurtosis_crest_decrease_match_definitions():
    rng = np.random.default_rng(0)
    sr = 16000
    t = np.arange(sr) / sr
    x = np.sin(2 * np.pi * 700 * t) + 0.3 * rng.standard_normal(sr)
    mag = _mag(x)
    f = np.fft.rfftfreq(1024, 1 / sr)[:, None]
    p = mag / mag.sum(0)
    mu = (f * p).sum(0)
    sd = np.sqrt((((f - mu) ** 2) * p).sum(0))
    kurt = (((f - mu) ** 4) * p).sum(0) / sd ** 4 - 3
    got = frequency_domain.extract_spectral_kurtosis(x, sr, frame_size=1024, hop_length=256)['spectral_kurtosis']
    np.testing.assert_allclose(got, kurt, rtol=1e-9)
    crest = frequency_domain.extract_spectral_crest(x, sr, frame_size=1024, hop_length=256)['spectral_crest']
    np.testing.assert_allclose(crest, mag.max(0) / mag.mean(0), rtol=1e-9)
    # pyACA FeatureSpectralDecrease
    k = np.arange(1, mag.shape[0])[:, None]
    dec = ((mag[1:] - mag[0]) / k).sum(0) / mag[1:].sum(0)
    got = frequency_domain.extract_spectral_decrease(x, sr, frame_size=1024, hop_length=256)['spectral_decrease']
    np.testing.assert_allclose(got, dec, rtol=1e-9)


def test_spectral_shape_extremes():
    sr = 16000
    tone = np.sin(2 * np.pi * 1000 * np.arange(sr) / sr)
    noise = np.random.default_rng(1).standard_normal(sr)
    kt = frequency_domain.extract_spectral_kurtosis(tone, sr)['spectral_kurtosis']
    kn = frequency_domain.extract_spectral_kurtosis(noise, sr)['spectral_kurtosis']
    # White noise is close to a uniform distribution over frequency (excess kurtosis -1.2);
    # a tone concentrates the magnitude around one frequency
    assert abs(np.median(kn) + 1.2) < 0.1 and np.median(kt) > np.median(kn) + 0.5
    ct = frequency_domain.extract_spectral_crest(tone, sr)['spectral_crest']
    cn = frequency_domain.extract_spectral_crest(noise, sr)['spectral_crest']
    assert np.median(ct) > 50 * np.median(cn)
    silent = np.zeros(4096)
    assert np.isnan(frequency_domain.extract_spectral_kurtosis(silent, sr)['spectral_kurtosis']).all()
    assert (frequency_domain.extract_spectral_crest(silent, sr)['spectral_crest'] == 1).all()
    assert (frequency_domain.extract_spectral_decrease(silent, sr)['spectral_decrease'] == 0).all()
