"""
Feature selection and dimensionality reduction utilities for AFX.
"""
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif


def pca_reducer(
    features: Dict[str, np.ndarray],
    n_components: int = 2,
    feature_keys: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, PCA]:
    """
    Reduce dimensionality of selected features using PCA.
    Args:
        features: Dict of feature arrays (flattened or 2D)
        n_components: Number of PCA components
        feature_keys: List of feature keys to include (default: all)
    Returns:
        Tuple of (reduced_features, fitted PCA object)
    """
    if feature_keys is None:
        feature_keys = list(features.keys())
    X = np.concatenate([features[k].reshape(-1, 1) if features[k].ndim == 1 else features[k].T for k in feature_keys], axis=1)
    pca = PCA(n_components=n_components)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca

def mutual_info_selector(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 10
    ) -> List[int]:
    """
    Select top-k features based on mutual information with target.
    Args:
        X: Feature matrix (samples x features)
        y: Target labels
        k: Number of features to select
    Returns:
        List of selected feature indices
    """
    mi = mutual_info_classif(X, y)
    selected = np.argsort(mi)[-k:][::-1]
    return selected.tolist()

def correlation_selector(
    X: np.ndarray,
    threshold: float = 0.95
    ) -> List[int]:
    """
    Select features by removing those highly correlated with others.
    Args:
        X: Feature matrix (samples x features)
        threshold: Correlation threshold for removal
    Returns:
        List of selected feature indices
    """
    corr = np.corrcoef(X, rowvar=False)
    n = corr.shape[0]
    to_remove = set()
    for i in range(n):
        for j in range(i+1, n):
            if abs(corr[i, j]) > threshold:
                to_remove.add(j)
    selected = [i for i in range(n) if i not in to_remove]
    return selected
