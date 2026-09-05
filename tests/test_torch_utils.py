"""
Unit tests for AFX.utils.torch_utils.

The conversion tests skip cleanly when torch is not installed; the
ImportError-message test runs only when torch is absent.
"""
import importlib.util

import numpy as np
import pytest

from AFX.utils.torch_utils import to_torch

TORCH_AVAILABLE = importlib.util.find_spec('torch') is not None


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='torch not installed')
def test_to_torch_converts_to_float32():
    import torch
    features = {'a': np.arange(5, dtype=np.float64), 'b': np.ones((2, 3))}
    out = to_torch(features)
    assert set(out.keys()) == {'a', 'b'}
    for v in out.values():
        assert isinstance(v, torch.Tensor)
        assert v.dtype == torch.float32


@pytest.mark.skipif(not TORCH_AVAILABLE, reason='torch not installed')
def test_to_torch_custom_dtype():
    import torch
    features = {'a': np.arange(5, dtype=np.float32)}
    out = to_torch(features, dtype=torch.float64)
    assert out['a'].dtype == torch.float64


@pytest.mark.skipif(TORCH_AVAILABLE, reason='torch is installed')
def test_to_torch_raises_helpful_import_error_without_torch():
    with pytest.raises(ImportError, match='PyTorch is not installed'):
        to_torch({'a': np.zeros(3)})
