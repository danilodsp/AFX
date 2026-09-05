"""
Torch compatibility utilities (for future extension).
"""
from typing import Dict, Optional

import numpy as np


def to_torch(features: Dict[str, np.ndarray], dtype=None):
    """
    Convert feature dict to torch.Tensor (if torch is available).

    Args:
        features: Dict of numpy arrays.
        dtype: Target torch dtype. If None, defaults to torch.float32.
    Returns:
        Dict mapping the same keys to torch.Tensors.
    Raises:
        ImportError: If PyTorch is not installed.

    Notes:
        Tensors are created with ``torch.as_tensor(v, dtype=...)``. When the
        input dtype differs from the target dtype (e.g. float64 numpy arrays
        converted to float32 tensors), the data is copied rather than
        aliased — so there are no memory-sharing surprises. Inputs already
        matching the target dtype may share memory with the source array.
    """
    try:
        import torch
    except ImportError as e:
        raise ImportError(
            'PyTorch is not installed. Install it with `pip install torch` '
            'to use AFX torch utilities.'
        ) from e
    if dtype is None:
        dtype = torch.float32
    return {k: torch.as_tensor(v, dtype=dtype) for k, v in features.items()}
