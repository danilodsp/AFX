"""
YIN fundamental frequency estimation.

de Cheveigné and Kawahara (2002), "YIN, a fundamental frequency estimator
for speech and music", JASA 111(4).
"""
import numpy as np

from AFX.methods.autocorrelation import autocorrelation


def _difference_function(frames: np.ndarray, tau_max: int) -> np.ndarray:
    """
    YIN difference function d(tau), tau = 0..tau_max, for each row of ``frames``.

        d(tau) = sum_{j=0}^{W-tau-1} (x[j] - x[j+tau])^2 = E1(tau) + E2(tau) - 2 r(tau)

    E1 and E2 are the energies of the two overlapping segments, from one
    cumulative sum, and r the FFT autocorrelation: O(W log W) per frame
    instead of O(W * tau_max).
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    w = frames.shape[1]
    cum = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    taus = np.arange(tau_max + 1)
    e1 = cum[:, w - taus]
    e2 = cum[:, w:w + 1] - cum[:, taus]
    d = e1 + e2 - 2.0 * autocorrelation(frames, tau_max + 1)
    return np.maximum(d, 0.0)


def _cmnd(d: np.ndarray) -> np.ndarray:
    """Cumulative mean normalized difference d'(tau) = d(tau) tau / sum_{1..tau} d; d'(0) = 1."""
    cmnd = np.ones_like(d)
    cum = np.cumsum(d[:, 1:], axis=1)
    taus = np.arange(1, d.shape[1])
    np.divide(d[:, 1:] * taus, cum, out=cmnd[:, 1:], where=cum > 0)
    return cmnd


def yin(y: np.ndarray, frame_length: int, hop_length: int, fmin: float = 50.0,
        fmax: float = 2000.0, sr: int = 22050, threshold: float = 0.1) -> np.ndarray:
    """
    Frame-wise f0 in Hz, 0 for frames judged unvoiced.

    Frames are taken without padding. The lag search runs over
    [sr / fmax, min(sr / fmin, frame_length / 2)]. In each frame the first
    dip of the cumulative mean normalized difference below ``threshold`` is
    followed to its local minimum (steps 3-4 of the paper); without such a
    dip the global minimum is used if it is below 0.5. The lag is refined by
    parabolic interpolation, and estimates outside [fmin, fmax] are set to 0.

    Returns an array of shape (n_frames,); a signal shorter than one frame
    gives [0.0].
    """
    y = np.asarray(y, dtype=np.float64)
    if len(y) < frame_length:
        return np.array([0.0])

    n_frames = 1 + (len(y) - frame_length) // hop_length
    tau_min = int(sr / fmax) if fmax > 0 else 1
    tau_max = min(int(sr / fmin) if fmin > 0 else frame_length // 2, frame_length // 2)
    if tau_min >= tau_max:
        return np.zeros(n_frames)

    frames = np.lib.stride_tricks.sliding_window_view(y, frame_length)[::hop_length]
    cmnd = _cmnd(_difference_function(frames, tau_max))
    rows = np.arange(n_frames)

    # First lag in [tau_min, tau_max) below the threshold...
    search = cmnd[:, tau_min:tau_max]
    below = search < threshold
    found = below.any(axis=1)
    start = tau_min + np.argmax(below, axis=1)
    # ...then walk down while the curve keeps decreasing (stop at tau_max - 1)
    lags = np.arange(tau_max)
    stops = np.ones((n_frames, tau_max), dtype=bool)
    stops[:, :-1] = cmnd[:, 1:tau_max] >= cmnd[:, :tau_max - 1]
    stops &= lags >= start[:, np.newaxis]
    dip = np.argmax(stops, axis=1)

    fallback = tau_min + np.argmin(cmnd[:, tau_min:tau_max + 1], axis=1)
    tau = np.where(found, dip, fallback).astype(np.float64)

    # Parabolic interpolation through (tau-1, tau, tau+1) when the vertex is a minimum
    t = tau.astype(np.intp)
    inner = (t > 0) & (t < tau_max)
    tc = np.clip(t, 1, tau_max - 1)
    a, b, c = cmnd[rows, tc - 1], cmnd[rows, tc], cmnd[rows, tc + 1]
    denom = a - 2 * b + c
    refine = inner & (denom > 0)
    offset = np.divide(0.5 * (a - c), denom, out=np.zeros_like(denom), where=refine)
    tau = tau + offset

    accept = found | (cmnd[rows, np.clip(tau.astype(np.intp), 0, tau_max)] < 0.5)
    pitch = np.where(accept & (tau > 0), sr / np.where(tau > 0, tau, 1.0), 0.0)
    pitch[(pitch < fmin) | (pitch > fmax)] = 0.0
    return pitch
