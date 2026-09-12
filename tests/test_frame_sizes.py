"""
Regression tests for non-default frame sizes.

Historically the framewise decorator did not forward frame_size/hop_length to
the wrapped functions, so any non-default frame size silently used the
defaults (wrong FFT sizes, all-zero pitch, crashes for frame_size < 2048).
These tests pin the fixed behavior.
"""
import numpy as np
import pytest

from AFX.extractors import (
    frequency_domain, harmonic_features, time_domain
)
from AFX.extract_all import extract_all_features, FEATURE_MAP


SR = 22050


def _tone(freq=440.0, duration=1.0, sr=SR):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    return np.sin(2 * np.pi * freq * t)


def expected_n_frames(n_samples, frame_size, hop_length):
    """Matches AFX.utils.frame_utils.frame_signal's framing convention."""
    return max(1, 1 + (n_samples - frame_size) // hop_length)


@pytest.mark.parametrize('frame_size,hop_length', [
    (512, 128), (1024, 256), (2048, 512), (4096, 1024),
])
def test_spectral_centroid_tracks_tone_at_any_frame_size(frame_size, hop_length):
    signal = _tone(440.0)
    result = frequency_domain.extract_spectral_centroid(
        signal, sr=SR, frame_size=frame_size, hop_length=hop_length
    )['spectral_centroid']
    assert np.all(np.isfinite(result))
    # Windowed pure tone: centroid must sit near 440 Hz at every frame size
    assert abs(np.median(result) - 440.0) < 60.0


@pytest.mark.parametrize('frame_size,hop_length', [
    (1024, 256), (2048, 512), (4096, 1024),
])
def test_pitch_nonzero_at_any_frame_size(frame_size, hop_length):
    signal = _tone(220.0)
    pitch = harmonic_features.extract_pitch(
        signal, sr=SR, frame_size=frame_size, hop_length=hop_length
    )['pitch']
    valid = pitch[pitch > 0]
    # Previously frame_size=1024 returned all zeros
    assert len(valid) > 0.5 * len(pitch)
    assert abs(np.median(valid) - 220.0) < 0.05 * 220.0


@pytest.mark.parametrize('frame_size,hop_length', [
    (1024, 256), (2048, 512), (4096, 1024),
])
def test_spectral_entropy_and_flatness_frame_counts(frame_size, hop_length):
    signal = _tone(440.0)
    # Previously frame_size < 2048 raised (np.zeros(-1)) and
    # frame_size > 2048 produced wrong-shaped output.
    n_exp = 1 + (len(signal) - frame_size) // hop_length
    ent = frequency_domain.extract_spectral_entropy(
        signal, sr=SR, frame_size=frame_size, hop_length=hop_length
    )['spectral_entropy']
    flat = frequency_domain.extract_spectral_flatness(
        signal, sr=SR, frame_size=frame_size, hop_length=hop_length
    )['spectral_flatness']
    assert ent.shape == (n_exp,)
    assert flat.shape == (n_exp,)
    assert np.all(np.isfinite(ent))
    assert np.all((flat >= 0) & (flat <= 1 + 1e-9))


def test_framewise_decorator_forwards_frame_size():
    """Decorated scalar features must produce one value per frame."""
    signal = _tone(440.0)
    frame_size, hop_length = 1000, 300  # deliberately non-default
    n_exp = expected_n_frames(len(signal), frame_size, hop_length)
    var = time_domain.extract_variance(
        signal, sr=SR, frame_size=frame_size, hop_length=hop_length
    )['variance']
    assert var.shape == (n_exp,)


def test_hnr_discriminates_harmonic_from_noise():
    rng = np.random.RandomState(0)
    tone = _tone(220.0)
    noise = rng.randn(len(tone))
    hnr_tone = harmonic_features.extract_hnr(tone, sr=SR)['hnr']
    hnr_noise = harmonic_features.extract_hnr(noise, sr=SR)['hnr']
    # A pure tone must score clearly higher than white noise, and white
    # noise must not come out strongly harmonic. (Previously the peak
    # search included lag 1, so *everything* scored ~50 dB.)
    assert np.median(hnr_tone) > np.median(hnr_noise) + 10.0
    assert np.median(hnr_noise) < 5.0


def test_extract_all_features_covers_default_config():
    """Every feature in the shipped config must be extracted (no silent skips)."""
    import json
    import os
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'AFX', 'config.json'
    )
    with open(config_path) as f:
        config = json.load(f)
    config['preserve_shape'] = True

    signal = _tone(440.0, duration=0.5)
    features = extract_all_features(signal, SR, config)

    missing = set(config['features']) - set(features)
    assert not missing, f'Features silently skipped: {sorted(missing)}'
    for name, arr in features.items():
        assert isinstance(arr, np.ndarray), name
        assert arr.size > 0, name


def test_extract_all_features_rejects_unknown_feature():
    config = {'features': {'definitely_not_a_feature': {}}}
    with pytest.raises(KeyError, match='definitely_not_a_feature'):
        extract_all_features(_tone(duration=0.1), SR, config)


def test_feature_map_covers_default_config():
    import json
    import os
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'AFX', 'config.json'
    )
    with open(config_path) as f:
        config = json.load(f)
    missing = set(config['features']) - set(FEATURE_MAP)
    assert not missing, f'config features missing from FEATURE_MAP: {sorted(missing)}'


def test_sample_entropy_rejects_huge_unframed_input():
    signal = np.random.RandomState(1).randn(30000)
    with pytest.raises(ValueError, match='O\\(N\\^2\\)'):
        time_domain.extract_sample_entropy(signal, sr=SR)


def test_sample_entropy_framed_is_finite_shape():
    signal = _tone(440.0, duration=0.25)
    se = time_domain.extract_sample_entropy(
        signal, sr=SR, frame_size=1024, hop_length=1024
    )['sample_entropy']
    n_exp = expected_n_frames(len(signal), 1024, 1024)
    assert se.shape == (n_exp,)
