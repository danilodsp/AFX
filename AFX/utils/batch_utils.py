"""
Batch utilities for stacking and concatenating aggregated feature vectors.
"""

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike


def features_to_vector(features: dict[str, np.ndarray], flatten: bool = True) -> np.ndarray:
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


def stack_feature_vectors(batch_features: list[dict[str, np.ndarray]], flatten: bool = True) -> np.ndarray:
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


def feature_names(features: Mapping[str, ArrayLike]) -> list[str]:
    """
    Column names for features_to_vector(features): ``name`` for a
    single value, ``name_0, name_1, ...`` (C order) for anything larger.
    """
    names = []
    for name, value in features.items():
        size = np.size(value)
        names.extend([name] if size == 1 else [f'{name}_{i}' for i in range(size)])
    return names


def flatten_features(features: Mapping[str, ArrayLike] | np.ndarray) -> tuple[list[str], np.ndarray]:
    """
    (column names, 1-D float vector) for a dict of aggregated features.

    The single array from aggregation='stack' gives names stack_0, stack_1, ...
    """
    if isinstance(features, np.ndarray):
        features = {'stack': features}
    if not features:
        return [], np.zeros(0)
    return feature_names(features), np.concatenate([np.ravel(v) for v in features.values()]).astype(np.float64)
