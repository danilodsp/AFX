"""
Frequency-domain feature extractors for audio signals.

All extractors are vectorized over a shared STFT (see AFX.methods.stft),
so multiple features computed on the same signal with the same framing
reuse a single STFT computation.
"""
from typing import Dict, Optional
import numpy as np

from AFX.methods.stft import stft, magnitude_spectrogram

__all__ = [
    'extract_spectral_centroid',
    'extract_spectral_bandwidth',
    'extract_spectral_rolloff',
    'extract_spectral_contrast',
    'extract_spectral_entropy',
    'extract_spectral_flatness',
    'extract_spectral_flux',
    'extract_spectral_skewness',
    'extract_spectral_slope',
]

_EPS = 1e-10


def _spectrum(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int],
    hop_length: Optional[int],
    center: bool,
):
    """
    Magnitude spectrum/spectrogram plus bin frequencies.

    Framed mode (frame_size and hop_length given): shared STFT of shape
    (n_freq, n_frames). Whole-signal mode (either is None): single FFT of
    the Hann-windowed full signal, shape (n_freq, 1).
    """
    if frame_size is None or hop_length is None:
        n = len(signal)
        if n == 0:
            raise ValueError('empty signal')
        windowed = signal * np.hanning(n)
        mag = np.abs(np.fft.rfft(windowed, n=n))[:, np.newaxis]
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        return mag, freqs
    stft_matrix = stft(
        signal, frame_size=frame_size, hop_length=hop_length,
        window='hann', center=center,
    )
    mag = magnitude_spectrogram(stft_matrix)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    return mag, freqs


def _maybe_metadata(result, key, values, sr, hop_length, frame_size,
                    center, return_metadata):
    if return_metadata:
        offset = (frame_size // 2) if (center and frame_size) else 0
        hop = hop_length if hop_length is not None else 1
        times = (np.arange(len(values)) * hop + offset) / sr
        result['metadata'] = {'times': times}
    return result


def extract_spectral_centroid(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral centroid (magnitude-weighted mean frequency).

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT (None = analyze whole signal)
        hop_length: Hop length between frames (None = analyze whole signal)
        center: If True, pad so frames are centered (librosa-style)
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_centroid' key, np.ndarray shape (n_frames,)
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = np.where(msum > 0, (freqs @ mag) / safe, 0.0)
    result = {'spectral_centroid': centroid}
    return _maybe_metadata(result, 'spectral_centroid', centroid, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_bandwidth(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral bandwidth (magnitude-weighted std of frequency
    around the spectral centroid).

    Args/returns: see extract_spectral_centroid.
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = (freqs @ mag) / safe
    dev2 = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
    bandwidth = np.where(
        msum > 0, np.sqrt((dev2 * mag).sum(axis=0) / safe), 0.0
    )
    result = {'spectral_bandwidth': bandwidth}
    return _maybe_metadata(result, 'spectral_bandwidth', bandwidth, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_rolloff(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    roll_percent: float = 0.85,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral rolloff: the frequency below which ``roll_percent``
    of the total spectral energy is contained (linearly interpolated).

    Args/returns: see extract_spectral_centroid, plus roll_percent.
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    power = mag ** 2
    cum = np.cumsum(power, axis=0)
    total = cum[-1, :]
    n_frames = mag.shape[1]
    rolloff = np.zeros(n_frames)

    nonzero = total > 0
    if np.any(nonzero):
        target = roll_percent * total
        # First bin where cumulative energy >= target
        idx = np.argmax(cum >= target[np.newaxis, :], axis=0)
        cols = np.arange(n_frames)
        # Linear interpolation between idx-1 and idx (where idx > 0)
        prev_idx = np.maximum(idx - 1, 0)
        e_before = cum[prev_idx, cols]
        e_after = cum[idx, cols]
        f_before = freqs[prev_idx]
        f_after = freqs[idx]
        denom = np.where(e_after > e_before, e_after - e_before, 1.0)
        alpha = np.clip((target - e_before) / denom, 0.0, 1.0)
        interp = f_before + alpha * (f_after - f_before)
        rolloff = np.where(idx > 0, interp, freqs[idx])
        rolloff = np.where(nonzero, rolloff, 0.0)

    result = {'spectral_rolloff': rolloff}
    return _maybe_metadata(result, 'spectral_rolloff', rolloff, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_skewness(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute the spectral skewness (third standardized spectral moment)."""
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = (freqs @ mag) / safe
    dev = freqs[:, np.newaxis] - centroid[np.newaxis, :]
    bandwidth = np.sqrt(((dev ** 2) * mag).sum(axis=0) / safe)
    safe_bw = np.where(bandwidth > 0, bandwidth, 1.0)
    skewness = ((dev / safe_bw) ** 3 * mag).sum(axis=0) / safe
    skewness = np.where((msum > 0) & (bandwidth > 0), skewness, 0.0)
    result = {'spectral_skewness': skewness}
    return _maybe_metadata(result, 'spectral_skewness', skewness, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_slope(
    signal: np.ndarray,
    sr: int,
    frame_size: Optional[int] = 2048,
    hop_length: Optional[int] = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute the spectral slope (linear regression of dB magnitude on Hz)."""
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    mag_db = 20.0 * np.log10(mag + _EPS)
    x = freqs - freqs.mean()
    denom = np.sum(x ** 2)
    if denom > 0:
        # sum(x - x_mean) == 0, so centering y is unnecessary
        slope = (x @ mag_db) / denom
    else:
        slope = np.zeros(mag.shape[1])
    result = {'spectral_slope': slope}
    return _maybe_metadata(result, 'spectral_slope', slope, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_entropy(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral entropy of an audio signal.

    The magnitude spectrum of each frame is normalized to a probability
    distribution p(f); entropy is H = -sum(p log2 p). Higher for noise-like
    frames, lower for tonal frames.

    Returns:
        Dict with 'spectral_entropy' key, np.ndarray shape (n_frames,)
    """
    mag, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > _EPS, msum, 1.0)
    ps = mag / safe
    entropy = -np.sum(ps * np.log2(ps + _EPS), axis=0)
    entropy = np.where(msum > _EPS, entropy, 0.0)
    result = {'spectral_entropy': entropy}
    return _maybe_metadata(result, 'spectral_entropy', entropy, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_flatness(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral flatness of an audio signal: the ratio of the
    geometric mean to the arithmetic mean of the power spectrum per frame.
    Close to 1.0 for noise, close to 0.0 for tonal content.

    Returns:
        Dict with 'spectral_flatness' key, np.ndarray shape (n_frames,)
    """
    mag, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    power = np.maximum(mag ** 2, _EPS)
    geometric = np.exp(np.mean(np.log(power), axis=0))
    arithmetic = np.mean(power, axis=0)
    flatness = np.where(arithmetic > _EPS, geometric / arithmetic, 0.0)
    result = {'spectral_flatness': flatness}
    return _maybe_metadata(result, 'spectral_flatness', flatness, sr,
                           hop_length, frame_size, center, return_metadata)


def extract_spectral_contrast(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    center: bool = True,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral contrast of an audio signal.

    For each of n_bands+1 octave-spaced sub-bands, contrast is the dB
    difference between the mean of the top-quantile (peak) and
    bottom-quantile (valley) magnitudes in each frame.

    Returns:
        Dict with 'spectral_contrast' key, np.ndarray (n_bands+1, n_frames)
    """
    stft_matrix = stft(
        signal, frame_size=frame_size, hop_length=hop_length,
        window='hann', center=center, pad_mode='constant',
    )
    S_mag = magnitude_spectrogram(stft_matrix)
    n_frames = S_mag.shape[1]

    freq = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    # Calculate logarithmically spaced bands
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    nyquist = sr / 2.0
    if np.any(octa[:-1] >= nyquist):
        raise ValueError("Frequency band exceeds Nyquist. Reduce fmin or n_bands.")

    valley = np.zeros((n_bands + 1, n_frames))
    peak = np.zeros((n_bands + 1, n_frames))

    for k in range(n_bands + 1):
        f_low, f_high = octa[k], octa[k + 1]
        current_band = np.logical_and(freq >= f_low, freq <= f_high)
        idx = np.flatnonzero(current_band)
        if len(idx) == 0:
            continue
        if k > 0 and idx[0] > 0:
            current_band[idx[0] - 1] = True
            idx = np.flatnonzero(current_band)
        if k == n_bands:
            current_band[idx[-1] + 1:] = True
            idx = np.flatnonzero(current_band)

        sub_band = S_mag[current_band]
        if k < n_bands and len(sub_band) > 0:
            sub_band = sub_band[:-1]
        if len(sub_band) == 0:
            continue

        n_quantile = int(np.maximum(np.rint(quantile * len(sub_band)), 1))

        # Sort once per band (all frames at once) instead of per frame
        sorted_band = np.sort(sub_band, axis=0)
        valley[k] = sorted_band[:n_quantile].mean(axis=0)
        peak[k] = sorted_band[-n_quantile:].mean(axis=0)

    peak_db = 20 * np.log10(peak + _EPS)
    valley_db = 20 * np.log10(valley + _EPS)
    contrast = peak_db - valley_db

    result = {'spectral_contrast': contrast}
    if return_metadata:
        # With center=True, frame i is centered at i * hop_length / sr
        # (librosa frames_to_time convention)
        times = np.arange(n_frames) * hop_length / sr
        result['metadata'] = {'times': times}
    return result


def extract_spectral_flux(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral flux: the Euclidean distance between consecutive
    magnitude spectra.

    Returns:
        Dict with 'spectral_flux' key, np.ndarray shape (n_frames - 1,)
    """
    stft_matrix = stft(
        signal, frame_size=frame_size, hop_length=hop_length,
        window='hann', center=center, pad_mode='reflect',
    )
    magnitude = magnitude_spectrogram(stft_matrix)

    diffs = np.diff(magnitude, axis=1)
    flux = np.sqrt(np.sum(diffs ** 2, axis=0))

    result = {'spectral_flux': flux}
    if return_metadata:
        offset = frame_size / 2 if center else 0
        times = (np.arange(len(flux)) * hop_length + offset) / sr
        result['metadata'] = {'times': times}
    return result
