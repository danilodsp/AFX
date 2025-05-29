"""
Mel-scale filterbank construction and processing.
"""
import numpy as np
from typing import Optional


def hz_to_mel(hz: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert frequency in Hz to mel scale.

    Args:
        hz: Frequency in Hz
        htk: Use HTK formula if True, otherwise use Slaney formula (librosa default)

    Returns:
        Frequency in mel scale
    """
    hz = np.asarray(hz)

    if htk:
        # HTK formula: mel = 2595 * log10(1 + hz / 700)
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    else:
        # Slaney formula (librosa default with htk=False)
        # Linear below 1000 Hz, log above
        f_min = 0.0
        f_sp = 200.0 / 3  # ~66.67 Hz per mel

        # Frequencies where we switch from linear to log
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0

        # Log scale parameters
        logstep = 0.068752  # Exact value matching librosa

        # Apply piecewise conversion
        mel = np.zeros_like(hz)

        # Linear region
        linear_mask = hz <= min_log_hz
        mel[linear_mask] = (hz[linear_mask] - f_min) / f_sp

        # Log region
        log_mask = hz > min_log_hz
        mel[log_mask] = min_log_mel + np.log(hz[log_mask] / min_log_hz) / logstep

        return mel


def mel_to_hz(mel: np.ndarray, htk: bool = False) -> np.ndarray:
    """
    Convert frequency in mel scale to Hz.

    Args:
        mel: Frequency in mel scale
        htk: Use HTK formula if True, otherwise use Slaney formula (librosa default)

    Returns:
        Frequency in Hz
    """
    mel = np.asarray(mel)

    if htk:
        # HTK formula: hz = 700 * (10^(mel/2595) - 1)
        return 700.0 * (10.0**(mel / 2595.0) - 1.0)
    else:
        # Slaney formula (librosa default with htk=False)
        f_min = 0.0
        f_sp = 200.0 / 3  # ~66.67 Hz per mel

        # Frequencies where we switch from linear to log
        min_log_hz = 1000.0
        min_log_mel = (min_log_hz - f_min) / f_sp  # 15.0

        # Log scale parameters
        logstep = 0.068752  # Exact value matching librosa

        # Apply piecewise conversion
        hz = np.zeros_like(mel)

        # Linear region
        linear_mask = mel <= min_log_mel
        hz[linear_mask] = f_min + mel[linear_mask] * f_sp

        # Log region  
        log_mask = mel > min_log_mel
        hz[log_mask] = min_log_hz * np.exp(logstep * (mel[log_mask] - min_log_mel))

        return hz


def mel_frequencies(
    n_mels: int,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    sr: int = 22050,
    htk: bool = False
) -> np.ndarray:
    """
    Compute mel-scale frequencies.

    Args:
        n_mels: Number of mel bands
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sr/2)
        sr: Sample rate
        htk: Use HTK formula if True, otherwise use Slaney formula (librosa default)

    Returns:
        Array of mel frequencies in Hz
    """
    if fmax is None:
        fmax = sr / 2.0

    # Convert to mel scale
    mel_min = hz_to_mel(np.array([fmin]), htk=htk)[0]
    mel_max = hz_to_mel(np.array([fmax]), htk=htk)[0]

    # Create mel scale points
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)

    # Convert back to Hz
    return mel_to_hz(mel_points, htk=htk)


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int = 22050,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: str = 'slaney',
    htk: bool = False
) -> np.ndarray:
    """
    Create a mel-scale filterbank.

    Args:
        n_mels: Number of mel filters
        n_fft: Length of FFT
        sr: Sample rate
        fmin: Minimum frequency in Hz
        fmax: Maximum frequency in Hz (defaults to sr/2)
        norm: Normalization method ('slaney' or None)
        htk: Use HTK formula if True, otherwise use Slaney formula (librosa default)

    Returns:
        Mel filterbank matrix of shape (n_mels, n_freq) where n_freq = n_fft // 2 + 1
    """
    if fmax is None:
        fmax = sr / 2.0

    # Get mel frequencies
    mel_freqs = mel_frequencies(n_mels, fmin, fmax, sr, htk=htk)

    # Convert to FFT bin indices
    n_freq = n_fft // 2 + 1
    fftfreqs = np.fft.fftfreq(n_fft, 1.0 / sr)[:n_freq]

    # Initialize filterbank
    filterbank = np.zeros((n_mels, n_freq))

    # Create triangular filters
    for i in range(n_mels):
        # Get the three mel frequencies for this filter
        left = mel_freqs[i]
        center = mel_freqs[i + 1]
        right = mel_freqs[i + 2]

        # Create triangular filter
        for j, freq in enumerate(fftfreqs):
            if left <= freq <= center:
                # Rising edge
                if center != left:
                    filterbank[i, j] = (freq - left) / (center - left)
            elif center < freq <= right:
                # Falling edge
                if right != center:
                    filterbank[i, j] = (right - freq) / (right - center)

    # Apply normalization
    if norm == 'slaney':
        # Slaney normalization: normalize by the width of each filter
        enorm = 2.0 / (mel_freqs[2:] - mel_freqs[:-2])
        filterbank *= enorm[:, np.newaxis]

    return filterbank


def apply_mel_filterbank(
    spectrogram: np.ndarray,
    filterbank: np.ndarray
) -> np.ndarray:
    """
    Apply mel filterbank to a spectrogram.

    Args:
        spectrogram: Power or magnitude spectrogram of shape (n_freq, n_frames)
        filterbank: Mel filterbank of shape (n_mels, n_freq)

    Returns:
        Mel spectrogram of shape (n_mels, n_frames)
    """
    return np.dot(filterbank, spectrogram)


def log_mel_spectrogram(
    mel_spec: np.ndarray,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0
) -> np.ndarray:
    """
    Convert mel spectrogram to log scale.

    Args:
        mel_spec: Mel spectrogram
        amin: Minimum value to clamp to (avoids log(0))
        top_db: Maximum dB range (for dynamic range limiting)

    Returns:
        Log mel spectrogram in dB
    """
    # Clamp to minimum value
    mel_spec = np.maximum(mel_spec, amin)

    # Convert to dB
    log_mel = 10.0 * np.log10(mel_spec)

    # Apply dynamic range limiting if specified
    if top_db is not None:
        log_mel = np.maximum(log_mel, log_mel.max() - top_db)

    return log_mel