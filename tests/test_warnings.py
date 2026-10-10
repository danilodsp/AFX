"""
Every warning AFX emits is an AFXWarning, so users can filter AFX apart from NumPy and SciPy.
"""
import ast
import warnings
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import afx_audio as afx
from afx_audio._warnings import warn_deprecated
from afx_audio.methods.mel import mel_filterbank

SR = 8000
PACKAGE = Path(afx.__file__).parent


def _write_tones(folder, durations):
    folder.mkdir(exist_ok=True)
    for i, seconds in enumerate(durations):
        t = np.arange(int(seconds * SR)) / SR
        sf.write(folder / f'tone_{i}.wav', np.sin(2 * np.pi * 440 * t).astype(np.float32), SR)
    return folder


def test_hierarchy():
    assert issubclass(afx.AFXWarning, UserWarning)
    assert issubclass(afx.AFXDeprecationWarning, afx.AFXWarning)
    assert issubclass(afx.AFXDeprecationWarning, DeprecationWarning)


def test_empty_mel_filters():
    with pytest.warns(afx.AFXWarning, match='Empty filters'):
        mel_filterbank(sr=SR, n_fft=64, n_mels=128)


def test_sample_rate_mismatch():
    with pytest.warns(afx.AFXWarning, match="differs from the config's sample_rate"):
        afx.extract_all_features(np.zeros(4000), 16000, {'sample_rate': SR, 'features': ['zcr']})


def test_batch_failed_file(tmp_path):
    folder = _write_tones(tmp_path / 'audio', [0.25])
    (folder / 'broken.wav').write_bytes(b'not audio')
    with pytest.warns(afx.AFXWarning, match='1 file'):
        afx.extract_features_from_folder(str(folder), {'sample_rate': SR, 'features': ['zcr']},
                                         str(tmp_path / 'f.json'), verbose=False)


def test_batch_empty_folder(tmp_path):
    (tmp_path / 'empty').mkdir()
    with pytest.warns(afx.AFXWarning, match='no audio files'):
        afx.extract_features_from_folder(str(tmp_path / 'empty'), {'sample_rate': SR, 'features': ['zcr']},
                                         str(tmp_path / 'f.json'), verbose=False)


def test_batch_ragged_columns(tmp_path):
    folder = _write_tones(tmp_path / 'audio', [0.25, 0.5])
    config = {'sample_rate': SR, 'features': ['zcr'], 'aggregation': 'stack'}
    with pytest.warns(afx.AFXWarning, match='different numbers of values'):
        afx.extract_features_from_folder(str(folder), config, str(tmp_path / 'f.csv'), verbose=False)


def test_filter_by_category_leaves_numpy_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        warnings.filterwarnings('ignore', category=afx.AFXWarning)
        mel_filterbank(sr=SR, n_fft=64, n_mels=128)
        np.mean(np.array([]))
    assert {w.category for w in caught} == {RuntimeWarning}


def _deprecated_function():
    warn_deprecated('_deprecated_function is deprecated; use extract_all_features')


def test_deprecation_points_at_the_caller():
    with pytest.warns(afx.AFXDeprecationWarning, match='deprecated') as record:
        _deprecated_function()
    assert record[0].filename == __file__


def test_deprecation_hidden_outside_main_by_default():
    # The interpreter's default filters, as seen by code that is not __main__
    with warnings.catch_warnings(record=True) as caught:
        warnings.resetwarnings()
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        warnings.filterwarnings('default', category=DeprecationWarning, module='__main__')
        _deprecated_function()
    assert caught == []


def _warn_calls_without_category(path):
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'warn'
                and isinstance(node.func.value, ast.Name) and node.func.value.id == 'warnings'
                and len(node.args) < 2 and not any(k.arg == 'category' for k in node.keywords)):
            yield f'{path.relative_to(PACKAGE.parent)}:{node.lineno}'


def test_every_warn_call_names_a_category():
    missing = [site for path in sorted(PACKAGE.rglob('*.py')) for site in _warn_calls_without_category(path)]
    assert missing == []
