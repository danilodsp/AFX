"""
Linear autocorrelation via FFT.
"""
import numpy as np
from scipy import fft as sfft


def autocorrelation(signal: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Linear (non-circular) autocorrelation for lags 0..max_lag-1 along the last axis.

    Equivalent to ``np.correlate(x, x, 'full')[n-1:n-1+max_lag]`` for each
    row but O(N log N): the signal is zero-padded to at least N + max_lag,
    so the circular wrap of the FFT product never reaches the returned lags.
    Accepts a 1-D signal or a 2-D (n_frames, frame_len) stack.

    Values smaller than 1e-12 * acf[0] are set to exactly 0.

    Returns the raw lagged product sums; divide by ``n - lag`` for the
    unbiased estimate or by ``acf[..., 0]`` for the normalized one.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.shape[-1]
    if n == 0:
        return np.zeros(signal.shape[:-1] + (0,))
    n_lags = min(max_lag, n)
    nfft = sfft.next_fast_len(n + n_lags, real=True)
    f = sfft.rfft(signal, nfft, axis=-1)
    acf = sfft.irfft(f.real ** 2 + f.imag ** 2, nfft, axis=-1)[..., :n_lags]
    # FFT round-off leaves values around eps * acf[0] where the exact result
    # is 0 (e.g. lags longer than the non-zero part of a frame). Snap them to
    # 0 so sign-based statistics do not count noise.
    acf[np.abs(acf) <= 1e-12 * acf[..., :1]] = 0.0
    return acf
