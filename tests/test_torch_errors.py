"""
Input and configuration errors on the torch path: each says what to change.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

import afx_audio.torch.functional as F_afx  # noqa: E402
from afx_audio.torch import AFXFeatureModule  # noqa: E402


@pytest.mark.parametrize(('waveform', 'error', 'message'), [
    (np.zeros(1024), TypeError, r'expected a torch.Tensor waveform, got ndarray; use torch.as_tensor'),
    (torch.zeros(1024, dtype=torch.int16), TypeError, 'floating-point tensor, got torch.int16'),
    (torch.zeros(2, 2, 1024), ValueError, r'1-D \(time,\) or 2-D \(batch, time\).*\(2, 2, 1024\)'),
    (torch.zeros(0), ValueError, 'stft received an empty signal'),
])
def test_waveform_errors(waveform, error, message):
    with pytest.raises(error, match=message):
        F_afx.stft(waveform, frame_size=256, hop_length=128)


@pytest.mark.parametrize(('kwargs', 'message'), [
    ({'features': []}, 'non-empty dict or list of names'),
    ({'features': ['mfcc'], 'aggregation': 'median'}, "aggregation must be 'none', 'mean', 'std' or 'mean_std'"),
    ({'features': ['mfcc'], 'output': 'frame'}, "output must be 'dict' or 'vector'"),
    ({'features': ['mfcc'], 'coverage': True}, "coverage needs aggregation 'mean', 'std' or 'mean_std'"),
    ({'features': {'mfcc': {'frame_size': None}}}, r'mfcc: whole-clip analysis \(frame_size=None\)'),
])
def test_module_configuration_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        AFXFeatureModule(sample_rate=16000, **kwargs)


def test_names_need_an_aggregation():
    with pytest.raises(ValueError, match="depend on the input length when aggregation='none'"):
        AFXFeatureModule(['mfcc'], sample_rate=16000).get_feature_names_out()
