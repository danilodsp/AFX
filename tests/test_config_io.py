"""
Unit tests for config_loader and io modules.
"""
import os
import numpy as np
import pytest
from AFX.utils import config_loader
from AFX.io import io

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
