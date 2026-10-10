"""
Linear-prediction features: LPC coefficients and formant frequencies.
"""
from typing import Any

import numpy as np

from afx_audio.methods.lpc import lpc, lpc_roots
from afx_audio.utils.frame_utils import analysis_frames, frame_times
from afx_audio.utils.validation import validate_signal

__all__ = ['extract_lpc', 'extract_formants']


def _symmetric_window(name, n):
    if name == 'hamming':
        return np.hamming(n)
    if name == 'hann':
        return np.hanning(n)
    from scipy.signal import get_window

    return get_window(name, n, fftbins=False)


def _windowed_frames(signal, frame_size, hop_length, center, window, pre_emphasis):
    signal = validate_signal(signal).astype(np.float64, copy=False)
    if pre_emphasis:
        # Applied before framing, as in Praat and most formant trackers
        signal = np.append(signal[:1], signal[1:] - pre_emphasis * signal[:-1])
    frames = analysis_frames(signal, frame_size, hop_length, center)
    if window is not None:
        frames = frames * _symmetric_window(window, frames.shape[1])
    return frames


def _times(frames, sr, frame_size, hop_length, center):
    if sr is None:
        raise ValueError('sr is required when return_metadata=True')
    centered = center and frame_size is not None and hop_length is not None
    return frame_times(len(frames), sr, hop_length or 0, frames.shape[1], centered)


def extract_lpc(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    order: int = 12,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    window: str | None = 'hamming',
    pre_emphasis: float = 0.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Linear prediction coefficients a_1..a_p per frame, shape (order, n_frames).

    Autocorrelation method with Levinson-Durbin recursion on each
    ``window``-weighted frame (symmetric window, None for rectangular),
    after optional pre-emphasis x[n] - pre_emphasis * x[n-1]. A(z) =
    1 + a_1 z^-1 + ... + a_p z^-p, the sign convention of librosa.lpc
    (which uses Burg's method, so values differ slightly). Framing as in
    the time-domain extractors; silent frames give zeros, frames not
    longer than ``order`` give NaN.
    """
    frames = _windowed_frames(signal, frame_size, hop_length, center, window, pre_emphasis)
    if frames.shape[1] <= order:
        coefficients = np.full((order, len(frames)), np.nan)
    else:
        coefficients = lpc(frames, order)[0][:, 1:].T
    out: dict[str, Any] = {'lpc': coefficients}
    if return_metadata:
        out['metadata'] = {'times': _times(frames, sr, frame_size, hop_length, center)}
    return out


def extract_formants(
    signal: np.ndarray,
    sr: int,
    *,
    n_formants: int = 3,
    frame_size: int = 512,
    hop_length: int = 128,
    center: bool = False,
    order: int | None = None,
    pre_emphasis: float = 0.97,
    fmin: float = 90.0,
    max_bandwidth: float = 400.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Formant frequencies F1..Fn in Hz per frame from LPC roots, shape (n_formants, n_frames).

    Each pre-emphasized, Hamming-windowed frame is fitted with an LPC
    polynomial of ``order`` (default 2 + sr // 1000). Roots in the upper
    half plane give resonances at f = angle * sr / (2 pi) with bandwidth
    -sr / pi * ln|z|; those above ``fmin`` and narrower than
    ``max_bandwidth`` are sorted by frequency and the lowest n_formants
    kept. NaN where a frame has fewer candidates (silence, unvoiced
    sounds). The classic method of Markel and Gray (1976); for speech use
    frames of 20 to 30 ms and resample to about 10 kHz or 2 kHz per
    expected formant, since high orders also fit spurious peaks.
    """
    if order is None:
        order = 2 + int(sr) // 1000
    frames = _windowed_frames(signal, frame_size, hop_length, center, 'hamming', pre_emphasis)
    if frames.shape[1] <= order:
        formants = np.full((n_formants, len(frames)), np.nan)
        out: dict[str, Any] = {'formants': formants}
        if return_metadata:
            out['metadata'] = {'times': _times(frames, sr, frame_size, hop_length, center)}
        return out
    a, err = lpc(frames, order)
    roots = lpc_roots(a)
    freqs = np.angle(roots) * sr / (2 * np.pi)
    with np.errstate(divide='ignore'):
        bandwidths = -sr / np.pi * np.log(np.abs(roots))
    valid = (roots.imag > 0) & (freqs > fmin) & (bandwidths < max_bandwidth) & (err[:, None] > 0)
    freqs = np.sort(np.where(valid, freqs, np.inf), axis=1)[:, :n_formants]
    if freqs.shape[1] < n_formants:
        freqs = np.pad(freqs, ((0, 0), (0, n_formants - freqs.shape[1])), constant_values=np.inf)
    formants = np.where(np.isfinite(freqs), freqs, np.nan).T
    out = {'formants': formants}
    if return_metadata:
        out['metadata'] = {'times': _times(frames, sr, frame_size, hop_length, center)}
    return out
