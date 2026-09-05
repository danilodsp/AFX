"""
Gammatone filterbank implementation for GFCC extraction.
"""
from functools import lru_cache

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

@lru_cache(maxsize=8)
def gammatone_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 50.0,
    fmax: Optional[float] = None,
) -> np.ndarray:
    """
    Create a gammatone filterbank matrix (n_filters, n_fft//2+1).

    Results are cached (lru_cache) and returned read-only; copy before
    mutating.
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
    # Equivalent rectangular bandwidth (Glasberg & Moore) in Hz, per filter
    b_hz = 1.019 * ((cf / ear_q) + min_bw)
    # 4th-order gammatone magnitude response (vectorized over filters):
    # |H(f)| ∝ [1 + ((f - fc) / b)^2]^(-order/2)
    filterbank = (
        1.0 + ((freqs[np.newaxis, :] - cf[:, np.newaxis])
               / b_hz[:, np.newaxis]) ** 2
    ) ** (-order / 2.0)
    # Normalize filters to unit area
    filterbank /= np.maximum(filterbank.sum(axis=1, keepdims=True), 1e-10)
    filterbank.flags.writeable = False
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
