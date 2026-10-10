"""
Unit tests for config_loader and io modules.
"""
import numpy as np
import pytest

from AFX.io import io
from AFX.utils import config_loader


def test_load_config(tmp_path):
    config_data = {
        "sample_rate": 22050,
        "features": {"zcr": {"frame_size": 2048, "hop_length": 512}},
        "aggregation": "mean"
    }
    config_path = tmp_path / "config.json"
    with open(config_path, 'w') as f:
        import json
        json.dump(config_data, f)
    config = config_loader.load_config(str(config_path))
    assert config['sample_rate'] == 22050
    assert 'zcr' in config['features']

def test_load_audio(tmp_path):
    # Generate a short sine wave and save as wav
    import soundfile as sf
    sr = 22050
    t = np.linspace(0, 1, sr, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "test.wav"
    sf.write(wav_path, x, sr)
    signal, sample_rate = io.load_audio(str(wav_path), sr=sr)
    assert isinstance(signal, np.ndarray)
    assert sample_rate == sr
    assert signal.shape[0] == sr

def test_load_audio_stereo_downmix(tmp_path):
    import soundfile as sf
    sr = 8000
    n = sr // 2
    left = 0.5 * np.ones(n)
    right = -0.5 * np.ones(n)
    stereo = np.column_stack([left, right])
    wav_path = tmp_path / "stereo.wav"
    sf.write(wav_path, stereo, sr)
    signal, sample_rate = io.load_audio(str(wav_path), sr=sr)
    assert signal.ndim == 1
    assert signal.shape[0] == n
    assert sample_rate == sr
    # Mean of +0.5 and -0.5 channels is ~0
    assert np.allclose(signal, 0.0, atol=1e-4)
    assert signal.dtype == np.float32

def test_load_audio_resample(tmp_path):
    import soundfile as sf
    native_sr = 8000
    target_sr = 22050
    t = np.linspace(0, 1, native_sr, endpoint=False)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)
    wav_path = tmp_path / "native.wav"
    sf.write(wav_path, x, native_sr)
    signal, sample_rate = io.load_audio(str(wav_path), sr=target_sr)
    assert sample_rate == target_sr
    expected_len = int(len(x) * target_sr / native_sr)
    assert abs(signal.shape[0] - expected_len) <= 1
    assert signal.dtype == np.float32

def test_load_audio_sr_none_keeps_native(tmp_path):
    import soundfile as sf
    native_sr = 8000
    x = 0.1 * np.random.RandomState(0).randn(native_sr)
    wav_path = tmp_path / "native.wav"
    sf.write(wav_path, x, native_sr)
    signal, sample_rate = io.load_audio(str(wav_path), sr=None)
    assert sample_rate == native_sr
    assert signal.shape[0] == native_sr

def test_load_audio_empty_file_raises(tmp_path):
    import soundfile as sf
    wav_path = tmp_path / "empty.wav"
    sf.write(wav_path, np.zeros((0,)), 8000)
    with pytest.raises(ValueError, match='empty'):
        io.load_audio(str(wav_path))

def test_load_audio_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.wav"
    with pytest.raises((FileNotFoundError, RuntimeError)):
        io.load_audio(str(missing))


def _write(tmp_path, config):
    import json
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return str(path)


def test_load_config_rejects_unknown_feature(tmp_path):
    path = _write(tmp_path, {'sample_rate': 22050, 'features': {'mfc': {}}})
    with pytest.raises(ValueError, match="mfc"):
        config_loader.load_config(path)


def test_load_config_rejects_bad_aggregation_and_normalize(tmp_path):
    path = _write(tmp_path, {'sample_rate': 22050, 'features': {'zcr': {}},
                             'aggregation': 'median'})
    with pytest.raises(ValueError, match='aggregation'):
        config_loader.load_config(path)
    path = _write(tmp_path, {'sample_rate': 22050, 'features': {'zcr': {}},
                             'normalize': 'l2'})
    with pytest.raises(ValueError, match='normalize'):
        config_loader.load_config(path)


def test_load_config_rejects_non_dict_feature_params(tmp_path):
    path = _write(tmp_path, {'sample_rate': 22050, 'features': {'zcr': [1]}})
    with pytest.raises(ValueError, match='zcr'):
        config_loader.load_config(path)


def test_validate_config_accepts_presets():
    from AFX.presets import list_presets, load_preset
    for name in list_presets():
        config_loader.validate_config(load_preset(name))


def test_load_audio_offset_and_duration(tmp_path):
    import soundfile as sf
    sr = 8000
    x = np.arange(sr * 2, dtype=np.float32) / (sr * 2)
    path = tmp_path / 'ramp.wav'
    sf.write(path, x, sr, subtype='FLOAT')
    seg, rate = io.load_audio(str(path), sr=None, offset=0.5, duration=0.25)
    assert rate == sr and len(seg) == sr // 4
    np.testing.assert_allclose(seg, x[sr // 2: sr // 2 + sr // 4])
    tail, _ = io.load_audio(str(path), sr=None, offset=1.75)
    assert len(tail) == sr // 4
    with open(path, 'rb') as f:
        whole, _ = io.load_audio(f, sr=None)
    assert len(whole) == 2 * sr
    with pytest.raises(ValueError, match='No audio samples'):
        io.load_audio(str(path), sr=None, offset=5.0)
    with pytest.raises(ValueError):
        io.load_audio(str(path), offset=-1)


def test_load_audio_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        io.load_audio(str(tmp_path / 'nope.wav'))
