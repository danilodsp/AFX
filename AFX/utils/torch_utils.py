"""
Torch compatibility utilities (for future extension).
"""
from typing import Dict
import numpy as np

def to_torch(features: Dict[str, np.ndarray]):
    """
    Convert feature dict to torch.Tensor (if torch is available).
    """
    try:
        import torch
        return {k: torch.from_numpy(v) for k, v in features.items()}
    except ImportError:
        raise RuntimeError('PyTorch is not installed.')
