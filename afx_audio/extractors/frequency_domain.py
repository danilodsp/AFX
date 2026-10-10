"""
Spectral shape descriptors computed from the STFT magnitude.

With ``frame_size`` and ``hop_length`` set (the default) every extractor
returns one value per STFT frame, shape (n_frames,), from the shared and
memoized STFT in afx_audio.methods.stft (periodic Hann window, n_fft =
frame_size, zero padding of frame_size // 2 on both sides when
``center``). Centroid, bandwidth, rolloff, skewness, kurtosis, slope, crest,
decrease, entropy and flatness also accept ``frame_size=None`` or ``hop_length=None`` to use one
Hann-windowed FFT of the whole clip, shape (1,).
"""

from typing import Any

import numpy as np

from afx_audio.methods.stft import magnitude_spectrogram, stft
from afx_audio.utils.frame_utils import frame_times
from afx_audio.utils.validation import float64_signal

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
    'extract_spectral_kurtosis',
    'extract_spectral_crest',
    'extract_spectral_decrease',
]

_EPS = 1e-10


def _spectrum(
    signal: np.ndarray,
    sr: int,
    frame_size: int | None,
    hop_length: int | None,
    center: bool,
):
    """
    Magnitude spectrum/spectrogram plus bin frequencies.

    Framed mode (frame_size and hop_length given): shared STFT of shape
    (n_freq, n_frames). Whole-signal mode (either is None): single FFT of
    the Hann-windowed full signal, shape (n_freq, 1).
    """
    signal = float64_signal(signal)
    if frame_size is None or hop_length is None:
        n = len(signal)
        windowed = signal * _periodic_hann(n)
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


def _periodic_hann(n):
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)


def _power_to_db(x, amin=1e-10, top_db=80.0):
    """librosa.power_to_db with ref=1: 10 log10(max(x, amin)), floored top_db below the array maximum."""
    db = 10.0 * np.log10(np.maximum(x, amin))
    return np.maximum(db, db.max() - top_db)


def _maybe_metadata(result, key, values, sr, hop_length, frame_size,
                    center, return_metadata, n_samples=None):
    if return_metadata:
        if frame_size is None or hop_length is None:
            # Whole-signal mode: one frame centered on the signal
            times = np.array([n_samples / (2.0 * sr)])
        else:
            times = frame_times(len(values), sr, hop_length, frame_size, center)
        result['metadata'] = {'times': times}
    return result


def extract_spectral_centroid(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Magnitude-weighted mean frequency in Hz, as librosa.feature.spectral_centroid. 0 for a silent frame."""
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = np.where(msum > 0, (freqs @ mag) / safe, 0.0)
    result: dict[str, Any] = {'spectral_centroid': centroid}
    return _maybe_metadata(result, 'spectral_centroid', centroid, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_bandwidth(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Magnitude-weighted standard deviation of frequency around the centroid, in Hz (librosa, p=2)."""
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = (freqs @ mag) / safe
    dev2 = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
    bandwidth = np.where(
        msum > 0, np.sqrt((dev2 * mag).sum(axis=0) / safe), 0.0
    )
    result: dict[str, Any] = {'spectral_bandwidth': bandwidth}
    return _maybe_metadata(result, 'spectral_bandwidth', bandwidth, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_rolloff(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    roll_percent: float = 0.85,
    center: bool = True,
    power: float = 1.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Spectral rolloff: the lowest bin frequency at which the cumulative sum
    of |X|^power reaches ``roll_percent`` of the frame total.

    ``power=1`` sums magnitudes, as librosa.feature.spectral_rolloff does
    (identical output for the same framing); ``power=2`` sums energy.
    0 Hz for an all-zero frame.
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    cum = np.cumsum(mag if power == 1 else mag ** power, axis=0)
    reached = cum >= roll_percent * cum[-1:, :]
    rolloff = freqs[np.argmax(reached, axis=0)]

    result: dict[str, Any] = {'spectral_rolloff': rolloff}
    return _maybe_metadata(result, 'spectral_rolloff', rolloff, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_skewness(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Third standardized moment of frequency under the magnitude spectrum. NaN for a silent or single-bin frame."""
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = (freqs @ mag) / safe
    dev = freqs[:, np.newaxis] - centroid[np.newaxis, :]
    bandwidth = np.sqrt(((dev ** 2) * mag).sum(axis=0) / safe)
    safe_bw = np.where(bandwidth > 0, bandwidth, 1.0)
    z = dev / safe_bw
    skewness = (z * z * z * mag).sum(axis=0) / safe
    skewness = np.where((msum > 0) & (bandwidth > 0), skewness, np.nan)
    result: dict[str, Any] = {'spectral_skewness': skewness}
    return _maybe_metadata(result, 'spectral_skewness', skewness, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_slope(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Least-squares slope of 20 log10(|X| + 1e-10) against frequency, in dB per Hz.

    NaN for a silent frame, where the slope would only be that of the
    1e-10 floor, and for a single-bin spectrum.
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    mag_db = 20.0 * np.log10(mag + _EPS)
    x = freqs - freqs.mean()
    denom = np.sum(x ** 2)
    if denom > 0:
        # sum(x - x_mean) == 0, so centering y is unnecessary
        slope = np.where(mag.any(axis=0), (x @ mag_db) / denom, np.nan)
    else:
        slope = np.full(mag.shape[1], np.nan)
    result: dict[str, Any] = {'spectral_slope': slope}
    return _maybe_metadata(result, 'spectral_slope', slope, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_kurtosis(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    fisher: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Fourth standardized moment of frequency under the magnitude spectrum, minus 3 when ``fisher``.

    The kurtosis of the spectrum seen as a distribution over frequency
    (Peeters, 2004, "A large set of audio features for sound
    description"), not the time-domain spectral kurtosis used for fault
    detection. 0 (or 3 without ``fisher``) is the value of a Gaussian
    shape; NaN for a silent or single-bin frame.
    """
    mag, freqs = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > 0, msum, 1.0)
    centroid = (freqs @ mag) / safe
    dev2 = (freqs[:, np.newaxis] - centroid[np.newaxis, :]) ** 2
    var = (dev2 * mag).sum(axis=0) / safe
    defined = (msum > 0) & (var > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        kurt = np.where(defined, (dev2 * dev2 * mag).sum(axis=0) / safe / var ** 2, np.nan)
    if fisher:
        kurt = kurt - 3.0
    result: dict[str, Any] = {'spectral_kurtosis': kurt}
    return _maybe_metadata(result, 'spectral_kurtosis', kurt, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_crest(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Largest over mean magnitude of the spectrum, per frame.

    1 for a flat spectrum; a sinusoid gives about frame_size / 4 with the
    Hann window (its energy spreads over three bins). NaN for a silent frame.
    """
    mag, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    mean = mag.mean(axis=0)
    crest = np.divide(mag.max(axis=0), mean, out=np.full_like(mean, np.nan), where=mean > 0)
    result: dict[str, Any] = {'spectral_crest': crest}
    return _maybe_metadata(result, 'spectral_crest', crest, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_decrease(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Spectral decrease (Peeters, 2004): sum_k (a_k - a_1) / (k - 1) / sum_k a_k over bins k = 2..K.

    a_k is the magnitude of bin k, with a_1 the DC bin, as in the Timbre
    Toolbox and Lerch's pyACA. Negative when the spectrum rises above its
    first bin on average. NaN when bins 2..K carry no energy (silence) or
    there is a single bin.
    """
    a, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    if len(a) < 2:
        decrease = np.full(a.shape[1], np.nan)
    else:
        k = np.arange(1, len(a))[:, np.newaxis]
        num = ((a[1:] - a[:1]) / k).sum(axis=0)
        den = a[1:].sum(axis=0)
        decrease = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    result: dict[str, Any] = {'spectral_decrease': decrease}
    return _maybe_metadata(result, 'spectral_decrease', decrease, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_entropy(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Shannon entropy in bits of the magnitude spectrum normalized to sum 1.

    At most log2(frame_size // 2 + 1) for a flat spectrum, near 0 for a
    single tone. NaN for a silent frame (magnitude sum at most 1e-10),
    which has no distribution to normalize.
    """
    mag, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    msum = mag.sum(axis=0)
    safe = np.where(msum > _EPS, msum, 1.0)
    ps = mag / safe
    entropy = -np.sum(ps * np.log2(ps + _EPS), axis=0)
    entropy = np.where(msum > _EPS, entropy, np.nan)
    result: dict[str, Any] = {'spectral_entropy': entropy}
    return _maybe_metadata(result, 'spectral_entropy', entropy, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_flatness(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Geometric over arithmetic mean of the power spectrum (Wiener entropy).

    Near 1 for white noise, near 0 for tones. Power is floored at 1e-10,
    as librosa.feature.spectral_flatness does with amin=1e-10, so silent
    frames give 1.
    """
    mag, _ = _spectrum(signal, sr, frame_size, hop_length, center)
    # The floor keeps both means >= 1e-10, so a silent frame is flat (1.0)
    power = np.maximum(mag.astype(np.float64) ** 2, _EPS)
    flatness = np.exp(np.mean(np.log(power), axis=0)) / np.mean(power, axis=0)
    result: dict[str, Any] = {'spectral_flatness': flatness}
    return _maybe_metadata(result, 'spectral_flatness', flatness, sr,
                           hop_length, frame_size, center, return_metadata,
                           len(signal))


def extract_spectral_contrast(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Spectral contrast (Jiang et al., 2002, Proc. IEEE ICME), shape (n_bands + 1, n_frames).

    The magnitude spectrum is split into octave bands starting at fmin
    (the first band is 0..fmin); in each band and frame, peak and valley
    are the means of the top and bottom ``quantile`` of magnitudes, and
    contrast is power_to_db(peak) - power_to_db(valley) with amin 1e-10 and
    top_db 80. This follows librosa.feature.spectral_contrast exactly,
    including its use of 10 log10 on magnitudes, so values are half the
    20 log10 magnitude ratio.
    """
    stft_matrix = stft(
        float64_signal(signal), frame_size=frame_size, hop_length=hop_length,
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

        # Quantile size from the band before its top bin is dropped, as librosa does
        n_quantile = int(np.maximum(np.rint(quantile * np.count_nonzero(current_band)), 1))

        # Sort once per band (all frames at once) instead of per frame
        sorted_band = np.sort(sub_band, axis=0)
        valley[k] = sorted_band[:n_quantile].mean(axis=0)
        peak[k] = sorted_band[-n_quantile:].mean(axis=0)

    contrast = _power_to_db(peak) - _power_to_db(valley)

    result: dict[str, Any] = {'spectral_contrast': contrast}
    if return_metadata:
        result['metadata'] = {
            'times': frame_times(n_frames, sr, hop_length, frame_size, center)
        }
    return result


def extract_spectral_flux(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Euclidean distance between consecutive magnitude spectra, shape (n_frames - 1,).

    Centering pads by reflection here (constant padding elsewhere). A clip
    with a single frame gives [0.0]. flux[i] compares frames i and i + 1 and
    carries the time of frame i + 1.
    """
    stft_matrix = stft(
        float64_signal(signal), frame_size=frame_size, hop_length=hop_length,
        window='hann', center=center, pad_mode='reflect',
    )
    magnitude = magnitude_spectrogram(stft_matrix)

    if magnitude.shape[1] < 2:
        # A single frame has no successor; report no change rather than an empty array
        flux = np.zeros(1)
    else:
        flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))

    result: dict[str, Any] = {'spectral_flux': flux}
    if return_metadata:
        # flux[i] compares frames i and i+1; report the time of frame i+1
        times = frame_times(magnitude.shape[1], sr, hop_length, frame_size, center)[-len(flux):]
        result['metadata'] = {'times': times}
    return result
