"""
Delta computation for feature sequences using finite difference method.
"""
import numpy as np
from scipy.ndimage import convolve1d


def delta_kernel(width: int) -> np.ndarray:
    """
    Create a delta kernel for finite difference computation.

    Args:
        width: Kernel width (must be odd and >= 3)

    Returns:
        Delta kernel weights of shape (width,)

    The kernel is computed:
    w[n] = n / (2 * sum(i^2 for i in range(1, (width+1)//2 + 1)))
    where n ranges from -(width-1)//2 to (width-1)//2
    """
    if width < 3:
        raise ValueError("Width must be at least 3")
    if width % 2 == 0:
        raise ValueError("Width must be odd")

    N = (width - 1) // 2
    indices = np.arange(-N, N + 1)

    # 2 * sum(i^2 for i in range(1, N+1))
    denominator = 2 * sum(i**2 for i in range(1, N + 1))
    if denominator == 0:
        # This happens when width == 1, but we already check width >= 3
        return np.zeros(width)

    # Compute kernel weights
    kernel = indices.astype(float) / denominator
    return kernel


def compute_delta(
    features: np.ndarray,
    width: int = 9,
    order: int = 1
) -> np.ndarray:
    """
    Compute delta (or delta-delta) features using finite difference method.

    Args:
        features: Feature matrix of shape (n_features, n_frames)
        width: Width of the delta kernel (must be odd and >= 3)
        order: Order of delta computation (1 or 2)

    Returns:
        Delta features of same shape as input

    Notes:
        - For order=1: computes first derivative (delta)
        - For order=2: computes second derivative (delta-delta)
        - Uses edge padding for boundary handling
    """
    if features.ndim != 2:
        raise ValueError("Features must be 2D (n_features, n_frames)")

    n_features, n_frames = features.shape

    # Handle edge case of insufficient frames
    if width >= n_frames:
        width = max(3, min(width, n_frames))
        if width % 2 == 0:
            width -= 1

    if width < 3:
        # Return zeros if not enough frames
        return np.zeros_like(features)

    # Get delta kernel
    kernel = delta_kernel(width)

    # Apply delta computation to each feature dimension
    delta_features = np.zeros_like(features)

    # Half width for padding
    N = (width - 1) // 2

    for i in range(n_features):
        signal = features[i]

        # Simple edge padding approach
        padded = np.pad(signal, N, mode='edge')

        # Apply convolution
        for t in range(n_frames):
            delta_features[i, t] = np.sum(kernel * padded[t:t+width])

        # For second order, apply delta again
        if order == 2:
            first_order = delta_features[i].copy()
            padded_first = np.pad(first_order, N, mode='edge')
            
            for t in range(n_frames):
                delta_features[i, t] = np.sum(kernel * padded_first[t:t+width])

    return delta_features