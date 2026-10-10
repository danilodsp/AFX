"""
Mel scale and triangular mel filterbanks, as in librosa (Slaney's Auditory Toolbox by default, HTK optional).
"""

import warnings

import numpy as np

# Slaney mel scale: linear below 1 kHz at 200/3 Hz per mel, logarithmic above
_F_SP = 200.0 / 3
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(hz: np.ndarray, htk: bool = False) -> np.ndarray:
    """Hz to mel: Slaney scale by default, 2595 log10(1 + f / 700) with ``htk``."""
    hz = np.asarray(hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    with np.errstate(divide='ignore'):
        log_part = _MIN_LOG_MEL + np.log(hz / _MIN_LOG_HZ) / _LOGSTEP
    return np.where(hz <= _MIN_LOG_HZ, hz / _F_SP, log_part)


def mel_to_hz(mel: np.ndarray, htk: bool = False) -> np.ndarray:
    """Inverse of hz_to_mel."""
    mel = np.asarray(mel, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    with np.errstate(over='ignore'):
        log_part = _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel - _MIN_LOG_MEL))
    return np.where(mel <= _MIN_LOG_MEL, mel * _F_SP, log_part)


def mel_frequencies(
    n_mels: int,
    fmin: float = 0.0,
    fmax: float | None = None,
    sr: int = 22050,
    htk: bool = False
) -> np.ndarray:
    """
    n_mels + 2 frequencies in Hz, equally spaced in mel from fmin to fmax (default sr / 2).

    These are the filter edges: left edge, the n_mels centers, right edge.
    librosa.mel_frequencies(n_mels + 2, ...) returns the same values.
    """
    if fmax is None:
        fmax = sr / 2.0
    mel_min, mel_max = hz_to_mel(np.array([fmin, fmax]), htk=htk)
    return mel_to_hz(np.linspace(mel_min, mel_max, n_mels + 2), htk=htk)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int = 22050,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str | None = 'slaney',
    htk: bool = False
) -> np.ndarray:
    """
    Triangular mel filterbank, shape (n_mels, n_fft // 2 + 1), equal to librosa.filters.mel.

    Filter m rises from edge m to center m + 1 and falls to edge m + 2 of
    mel_frequencies. ``norm='slaney'`` divides each filter by half its
    width in Hz so every band has the same area; ``norm=None`` leaves a
    peak of 1. Warns, like librosa, when a filter falls between FFT bins
    and is all zero.
    """
    if fmax is None:
        fmax = sr / 2.0
    mel_freqs = mel_frequencies(n_mels, fmin, fmax, sr, htk=htk)
    # rfftfreq, not fftfreq: the Nyquist bin must be +sr/2 or the top filter is cut
    fftfreqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    fdiff = np.diff(mel_freqs)
    ramps = mel_freqs[:, np.newaxis] - fftfreqs[np.newaxis, :]
    lower = -ramps[:-2] / np.maximum(fdiff[:-1][:, np.newaxis], 1e-12)
    upper = ramps[2:] / np.maximum(fdiff[1:][:, np.newaxis], 1e-12)
    filterbank = np.maximum(0.0, np.minimum(lower, upper))
    if not np.all((mel_freqs[:-2] == 0) | (filterbank.max(axis=1) > 0)):
        warnings.warn(
            f'Empty filters in the mel filterbank (n_mels={n_mels}, n_fft={n_fft}, sr={sr}): some bands fall '
            'between FFT bins and stay zero. Use fewer mels or a larger n_fft.',
            stacklevel=2,
        )
    if norm == 'slaney':
        filterbank *= (2.0 / (mel_freqs[2:] - mel_freqs[:-2]))[:, np.newaxis]
    return filterbank


def apply_mel_filterbank(spectrogram: np.ndarray, filterbank: np.ndarray) -> np.ndarray:
    """(n_mels, n_freq) @ (n_freq, n_frames) -> (n_mels, n_frames)."""
    return np.dot(filterbank, spectrogram)


def log_mel_spectrogram(
    mel_spec: np.ndarray,
    amin: float = 1e-10,
    top_db: float | None = 80.0
) -> np.ndarray:
    """10 log10(max(S, amin)), floored ``top_db`` below the maximum (librosa.power_to_db with ref=1)."""
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, amin))
    if top_db is not None:
        log_mel = np.maximum(log_mel, np.max(log_mel) - top_db)
    return log_mel
