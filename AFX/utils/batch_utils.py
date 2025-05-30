"""
Batch utilities for stacking and concatenating aggregated feature vectors.
"""
from typing import List, Dict
import numpy as np


def features_to_vector(features: Dict[str, np.ndarray], flatten: bool = True) -> np.ndarray:
    """
    Concatenate all (optionally flattened) features into a single 1D vector.
    Args:
        features: Dict of feature arrays (aggregated per file)
        flatten: Whether to flatten each feature before concatenation
    Returns:
        1D numpy array (all features concatenated)
    """
    vecs = []
    for v in features.values():
        arr = v.flatten() if flatten else v
        vecs.append(arr)
    return np.concatenate(vecs)


def stack_feature_vectors(batch_features: List[Dict[str, np.ndarray]], flatten: bool = True) -> np.ndarray:
    """
    Stack feature vectors for a batch of files into a 2D numpy array.
    Args:
        batch_features: List of feature dicts (one per file)
        flatten: Whether to flatten each feature before concatenation
    Returns:
        2D numpy array of shape (batch_size, total_features)
    """
    vectors = [features_to_vector(f, flatten=flatten) for f in batch_features]
    return np.stack(vectors, axis=0)
