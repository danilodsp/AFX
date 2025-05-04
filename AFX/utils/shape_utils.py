"""
Shape normalization and flattening utilities for features.
"""
from typing import Dict
import numpy as np

def normalize_feature_shape(feature: np.ndarray, mode: str = 'flatten') -> np.ndarray:
    """
    Normalize feature shape: flatten or keep dimensions.
    Args:
        feature: Input feature array
        mode: 'flatten' or 'keepdims'
    Returns:
        np.ndarray
    """
    if mode == 'flatten':
        return feature.flatten()
    elif mode == 'keepdims':
        return feature
    else:
        raise ValueError(f'Unknown mode: {mode}')

def normalize_features_dict(features: Dict[str, np.ndarray], mode: str = 'flatten') -> Dict[str, np.ndarray]:
    """
    Normalize all features in a dict.
    """
    return {k: normalize_feature_shape(v, mode=mode) for k, v in features.items()}
