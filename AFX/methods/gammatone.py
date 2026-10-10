"""
Gammatone filterbank on STFT bins for GFCC.
"""
from functools import lru_cache

import numpy as np

# Glasberg and Moore (1990): ERB(f) = f / EAR_Q + MIN_BW
EAR_Q = 9.26449
MIN_BW = 24.7


def erb_space(fmin: float, fmax: float, n_filters: int) -> np.ndarray:
    """``n_filters`` center frequencies in Hz equally spaced on the ERB-rate scale from fmin to fmax."""
    erb_low = np.log(fmin / EAR_Q + MIN_BW)
    erb_high = np.log(fmax / EAR_Q + MIN_BW)
    return np.exp(np.linspace(erb_low, erb_high, n_filters)) * EAR_Q - MIN_BW * EAR_Q


@lru_cache(maxsize=8)
def gammatone_filterbank(
    n_filters: int,
    n_fft: int,
    sr: int,
    fmin: float = 50.0,
    fmax: float | None = None,
) -> np.ndarray:
    """
    Gammatone weights on the rfft bins, shape (n_filters, n_fft // 2 + 1).

    Filter i is the magnitude response of a 4th-order gammatone,
    (1 + ((f - fc_i) / b_i)^2)^-2 with b_i = 1.019 ERB(fc_i), normalized to
    unit sum. Centers come from erb_space(fmin, fmax). Cached and
    read-only; copy before modifying.
    """
    if fmax is None:
        fmax = sr / 2.0
    cf = erb_space(fmin, fmax, n_filters)
    freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    order = 4
    bandwidth = 1.019 * (cf / EAR_Q + MIN_BW)
    filterbank = (
        1.0 + ((freqs[np.newaxis, :] - cf[:, np.newaxis]) / bandwidth[:, np.newaxis]) ** 2
    ) ** (-order / 2.0)
    filterbank /= np.maximum(filterbank.sum(axis=1, keepdims=True), 1e-10)
    filterbank.flags.writeable = False
    return filterbank


def apply_gammatone_filterbank(power_spec: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    """(n_filters, n_freq) @ (n_freq, n_frames) -> (n_filters, n_frames)."""
    return np.dot(filterbank, power_spec)
