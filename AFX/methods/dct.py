"""
Discrete Cosine Transform (DCT) utilities.
"""
import numpy as np
from scipy.fftpack import dct


def dct_ii(
    x: np.ndarray,
    norm: str = 'ortho',
    axis: int = -1
) -> np.ndarray:
    """
    Compute DCT-II (Type 2 DCT) of input array.

    This is a wrapper around scipy.fftpack.dct for consistency.

    Args:
        x: Input array
        norm: Normalization ('ortho' for orthogonal normalization)
        axis: Axis along which to compute DCT

    Returns:
        DCT-II transformed array
    """
    return dct(x, type=2, norm=norm, axis=axis)


def extract_mfcc_coefficients(
    log_mel_spec: np.ndarray,
    n_mfcc: int = 13,
    norm: str = 'ortho'
) -> np.ndarray:
    """
    Extract MFCC coefficients from log mel spectrogram using DCT-II.

    Args:
        log_mel_spec: Log mel spectrogram of shape (n_mels, n_frames)
        n_mfcc: Number of MFCC coefficients to extract
        norm: Normalization for DCT

    Returns:
        MFCC coefficients of shape (n_mfcc, n_frames)
    """
    # Apply DCT-II along mel dimension (axis 0)
    mfcc_full = dct_ii(log_mel_spec, norm=norm, axis=0)

    # Extract first n_mfcc coefficients
    return mfcc_full[:n_mfcc, :]