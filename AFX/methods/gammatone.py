"""
Gammatone filterbank implementation for GFCC extraction.
"""
import numpy as np
from typing import Optional

def erb_space(fmin: float, fmax: float, n_filters: int) -> np.ndarray:
    """
    Compute center frequencies spaced equally on the ERB scale.
    """
    # Glasberg & Moore (1990) ERB-rate scale
    ear_q = 9.26449
    min_bw = 24.7
    erb_low = np.log(fmin / ear_q + min_bw)
    erb_high = np.log(fmax / ear_q + min_bw)
    cf = np.exp(np.linspace(erb_low, erb_high, n_filters)) * ear_q - min_bw * ear_q
    return cf

def gammatone_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Create a gammatone filterbank matrix (n_filters, n_fft//2+1).
    """
    if fmax is None:
        fmax = sr / 2.0
    # Center frequencies on ERB scale
    cf = erb_space(fmin, fmax, n_filters)
    # Frequency bins
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    ear_q = 9.26449
    min_bw = 24.7
    order = 4
    filterbank = np.zeros((n_filters, len(freqs)))
    for i, fc in enumerate(cf):
        erb = ((fc / ear_q) + min_bw)
        b = 1.019 * 2 * np.pi * erb
        # Gammatone magnitude response (simplified)
        filterbank[i, :] = (freqs ** (order - 1)) / (
            1 + ((freqs - fc) / (b / (2 * np.pi))) ** 2
        )
    # Normalize filters to unit area
    filterbank /= np.maximum(filterbank.sum(axis=1, keepdims=True), 1e-10)
    return filterbank

def apply_gammatone_filterbank(
    power_spec: np.ndarray, filterbank: np.ndarray
) -> np.ndarray:
    """
    Apply gammatone filterbank to power spectrogram.
    Args:
        power_spec: (n_freqs, n_frames)
        filterbank: (n_filters, n_freqs)
    Returns:
        (n_filters, n_frames)
    """
    return np.dot(filterbank, power_spec)
