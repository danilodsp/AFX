"""
Linear autocorrelation via FFT.
"""
import numpy as np
from scipy.fft import next_fast_len


def autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Linear (non-circular) autocorrelation for lags 0..max_lag-1.

    Equivalent to ``np.correlate(signal, signal, 'full')[n-1:n-1+max_lag]``
    but O(N log N): the signal is zero-padded to at least 2N so the
    circular wrap of the FFT product never aliases into the returned lags.

    Returns the raw lagged product sums (biased by nothing); divide by
    ``n - lag`` for the unbiased estimate or by ``acf[0]`` for the
    normalized one.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)
    if n == 0:
        return np.zeros(0)
    nfft = next_fast_len(2 * n)
    f = np.fft.rfft(signal, nfft)
    acf = np.fft.irfft(f * np.conj(f), nfft)
    return acf[:min(max_lag, n)]
