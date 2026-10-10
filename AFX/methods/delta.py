"""
Delta (derivative) features over time.
"""
import numpy as np


def delta_kernel(width: int) -> np.ndarray:
    """
    Regression kernel n / (2 sum_{i=1}^{N} i^2) for n = -N..N, N = (width - 1) // 2.

    Correlating with it gives the interior of compute_delta(order=1).
    """
    if width < 3:
        raise ValueError("Width must be at least 3")
    if width % 2 == 0:
        raise ValueError("Width must be odd")

    half = (width - 1) // 2
    n = np.arange(-half, half + 1, dtype=np.float64)
    return n / (2.0 * np.sum(np.arange(1, half + 1) ** 2))


def compute_delta(
    features: np.ndarray,
    width: int = 9,
    order: int = 1
) -> np.ndarray:
    """
    Delta features along the last axis, same as librosa.feature.delta.

    A Savitzky-Golay filter of length ``width`` and polynomial order
    ``order`` returns the ``order``-th derivative; in the interior this is
    the regression formula of delta_kernel for order 1. Edges are fitted
    with the same polynomial (mode='interp') rather than padded.
    ``width`` is clipped to the largest odd value not above n_frames.

    Args:
        features: (n_features, n_frames)
        width: odd window length, at least 3
        order: 1 for delta, 2 for delta-delta
    """
    from scipy.signal import savgol_filter

    if features.ndim != 2:
        raise ValueError("Features must be 2D (n_features, n_frames)")
    if order not in (1, 2):
        raise ValueError(f"order must be 1 or 2, got {order}")
    n_frames = features.shape[1]
    if width >= n_frames:
        width = max(3, min(width, n_frames))
        if width % 2 == 0:
            width -= 1
    if width < 3 or width % 2 == 0:
        raise ValueError(f'width must be odd and at least 3, got {width}')
    if n_frames < width:
        raise ValueError(f'need at least {width} frames, got {n_frames}')
    return savgol_filter(features.astype(np.float64), width, polyorder=order, deriv=order, mode='interp', axis=-1)
