"""Output conventions shared by every built-in feature: dtype, framing and silent frames."""
import numpy as np
import pytest

import afx_audio

SR = 16000
FEATURES = sorted(afx_audio.FEATURE_MAP)
NOISE = np.random.default_rng(0).standard_normal(SR // 2) * 0.1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', FEATURES)
def test_output_is_float64(name, dtype):
    out = afx_audio.FEATURE_MAP[name](NOISE.astype(dtype), SR, return_metadata=True)
    assert out[name].dtype == np.float64
    assert np.asarray(out['metadata']['times']).dtype == np.float64


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_torch_output_follows_input_dtype(dtype_name):
    torch = pytest.importorskip('torch')
    from afx_audio.torch import functional as F_afx
    from afx_audio.torch.module import _TORCH_FEATURES

    dtype = getattr(torch, dtype_name)
    x = torch.as_tensor(NOISE, dtype=dtype)
    for name, func in _TORCH_FEATURES.items():
        assert func(x, SR).dtype == dtype, name
    assert F_afx.delta(F_afx.mfcc(x, SR)).dtype == dtype
