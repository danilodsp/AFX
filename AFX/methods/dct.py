"""
DCT-II for cepstral coefficients.
"""
import numpy as np
from scipy.fft import dct


def dct_ii(x: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
    """scipy.fft.dct with type 2."""
    return dct(x, type=2, norm=norm, axis=axis)


def extract_mfcc_coefficients(log_mel_spec: np.ndarray, n_mfcc: int = 13, norm: str = 'ortho') -> np.ndarray:
    """First ``n_mfcc`` rows of the DCT-II over the band axis of a (n_bands, n_frames) log spectrogram."""
    return dct_ii(log_mel_spec, norm=norm, axis=0)[:n_mfcc, :]
