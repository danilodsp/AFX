"""
Harmonic feature extractors: pitch (YIN), total harmonic distortion and
harmonics-to-noise ratio.
"""

from typing import Any

import numpy as np

from afx_audio.methods.autocorrelation import autocorrelation as _autocorr
from afx_audio.utils.frame_utils import analysis_frames, frame_times
from afx_audio.utils.pitch import yin
from afx_audio.utils.validation import validate_signal

_EPS = 1e-10

__all__ = [
    'extract_pitch',
    'extract_thd',
    'extract_hnr',
]


def extract_pitch(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    threshold: float = 0.1,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Fundamental frequency in Hz per frame by YIN, NaN for unvoiced frames.

    A frame is unvoiced when the cumulative mean normalized difference has
    no dip below ``threshold`` and its minimum is not below 0.5, or when
    the estimate falls outside [fmin, fmax] (see afx_audio.utils.pitch.yin,
    which returns 0 for these frames). With return_metadata=True,
    ``metadata['voiced']`` is the boolean voicing mask. The longest
    detectable period is frame_size // 2 samples, so fmin is effectively
    max(fmin, 2 sr / frame_size). Shape (n_frames,), frames unpadded.
    """
    f0 = yin(validate_signal(signal), frame_length=frame_size, hop_length=hop_length, fmin=fmin, fmax=fmax, sr=sr,
             threshold=threshold)
    voiced = f0 > 0
    result: dict[str, Any] = {'pitch': np.where(voiced, f0, np.nan)}
    if return_metadata:
        result['metadata'] = {'times': frame_times(len(f0), sr, hop_length, frame_size), 'voiced': voiced}
    return result


def extract_thd(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    n_harmonics: int = 4,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Total harmonic distortion sqrt(sum_h A_h^2) / A_1, as a ratio.

    A_1 is the largest non-DC bin of the unwindowed FFT, A_h the largest
    bin within +/-2 bins of h times its index, for h = 2..n_harmonics + 1
    below Nyquist. Assumes the strongest component is the fundamental, so
    it is meant for tonal signals (test tones, machinery hum). NaN for
    frames with fewer than 2 bins or no energy. Framing as in the
    time-domain extractors; shape (n_frames,).
    """
    frames = analysis_frames(signal, frame_size, hop_length, center)
    spectrum = np.abs(np.fft.rfft(frames, axis=1))
    n_bins = spectrum.shape[1]
    thd = np.full(len(frames), np.nan)
    if n_bins >= 2:
        rows = np.arange(len(frames))
        f0 = 1 + np.argmax(spectrum[:, 1:], axis=1)
        a1 = spectrum[rows, f0]
        harmonic_power = np.zeros(len(frames))
        offsets = np.arange(-2, 3)
        for h in range(2, n_harmonics + 2):
            harmonic_bin = f0 * h
            idx = harmonic_bin[:, np.newaxis] + offsets
            # Search window clipped at Nyquist, as with the slice [idx - 2, idx + 3)
            peak = np.where(idx < n_bins, spectrum[rows[:, np.newaxis], np.minimum(idx, n_bins - 1)], -np.inf).max(1)
            harmonic_power += np.where(harmonic_bin < n_bins, peak, 0.0) ** 2
        ok = a1 > 0
        thd[ok] = np.sqrt(harmonic_power[ok]) / (a1[ok] + _EPS)
    out: dict[str, Any] = {'thd': thd}
    if return_metadata:
        centered = center and frame_size is not None and hop_length is not None
        out['metadata'] = {'times': frame_times(len(frames), sr, hop_length or 0, frames.shape[1], centered)}
    return out


def extract_hnr(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Harmonics-to-noise ratio in dB per frame, 10 log10(r / (1 - r)).

    The dB mapping is that of Boersma (1993, Proc. Institute of Phonetic
    Sciences 17); the correlation differs from his window-corrected
    autocorrelation. r is the maximum over lags sr / fmax .. sr / fmin of the normalized
    cross-correlation of the two segments overlapping at each lag (the
    NCCF of Talkin, 1995, "A robust algorithm for pitch tracking"):

        r(tau) = sum_j x[j] x[j+tau] / sqrt(sum_j x[j]^2 * sum_j x[j+tau]^2)

    for j = 0..N-tau-1. It is 1 for a perfectly periodic frame. The biased
    estimate R(tau) / R(0) would cap r at (N - tau) / N, about 2 dB HNR for
    a pure tone with a 400-sample frame and a 160-sample period.

    r is clipped to (1e-10, 1 - 1e-10). NaN for silent frames (zero
    energy); low-level noise still gives a finite, low HNR. A clip
    shorter than one frame is zero-padded to one frame. Shape (n_frames,).
    """
    signal = validate_signal(signal).astype(np.float64, copy=False)
    n_samples = len(signal)
    n_frames = max(1, 1 + (n_samples - frame_size) // hop_length)
    if n_samples < frame_size:
        frames = np.pad(signal, (0, frame_size - n_samples))[np.newaxis, :]
    else:
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_length]

    # Restrict the peak search to plausible pitch lags. Over all lags,
    # r(1) ~ 1 for any smooth signal and HNR would be meaningless.
    lag_min = max(1, int(sr / fmax)) if fmax > 0 else 1
    lag_max = min(int(sr / fmin) if fmin > 0 else frame_size - 1, frame_size - 1)

    hnr = np.full(n_frames, np.nan)
    if lag_min < lag_max:
        lags = np.arange(lag_max + 1)
        # Energies of x[0:N-tau] and x[tau:N] for every lag from one cumulative sum
        cum = np.concatenate([np.zeros((n_frames, 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
        e_lead = cum[:, frame_size - lags]
        e_trail = cum[:, frame_size:] - cum[:, lags]
        r = _autocorr(frames, lag_max + 1) / np.sqrt(np.maximum(e_lead * e_trail, _EPS))
        r_max = np.clip(r[:, lag_min:lag_max + 1].max(axis=1), _EPS, 1.0 - _EPS)
        voiced = e_lead[:, 0] > 0
        hnr[voiced] = 10 * np.log10(r_max[voiced] / (1.0 - r_max[voiced]))
    result: dict[str, Any] = {'hnr': hnr}
    if return_metadata:
        result['metadata'] = {'times': frame_times(n_frames, sr, hop_length, frame_size)}
    return result
