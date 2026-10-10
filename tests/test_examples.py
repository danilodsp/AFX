import importlib.util
import os

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip('sklearn')

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(EXAMPLES, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_classify_folders(tmp_path, capsys):
    sr = 16000
    rng = np.random.default_rng(0)
    t = np.arange(sr // 2) / sr
    for label, make in [('tone', lambda: np.sin(2 * np.pi * rng.uniform(200, 400) * t)),
                        ('noise', lambda: rng.standard_normal(len(t)))]:
        (tmp_path / label).mkdir()
        for i in range(6):
            sf.write(tmp_path / label / f'{i}.wav', (0.3 * make()).astype(np.float32), sr)
    scores = _load('classify_folders').main([str(tmp_path), '--folds', '3'])
    assert scores.mean() > 0.9
    assert '12 files, 2 classes' in capsys.readouterr().out


def test_extract_and_plot(tmp_path, capsys):
    sr = 22050
    path = tmp_path / 'clip.wav'
    sf.write(path, (0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32), sr)
    assert _load('example_extract_and_plot').main([str(path), '--no-plot']) == 0
    assert 'mfcc per-frame shape' in capsys.readouterr().out
