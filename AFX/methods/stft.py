"""
Short-Time Fourier Transform (STFT) implementation.
"""
import numpy as np
from typing import Tuple


def stft(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop_length: int = 512,
    window: str = 'hann',
    center: bool = True
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.

    Args:
        signal: Input signal (1D array)
        frame_size: Length of each frame (n_fft)
        hop_length: Number of samples between adjacent frames
        window: Window function ('hann', 'hamming', 'blackman', or None)
        center: Whether to center frames around the signal

    Returns:
        Complex STFT matrix of shape (n_freq, n_frames) where n_freq = frame_size // 2 + 1
    """
    # Center padding if requested
    if center:
        pad_width = frame_size // 2
        signal = np.pad(signal, (pad_width, pad_width), mode='reflect')

    # Calculate number of frames
    n_frames = 1 + (len(signal) - frame_size) // hop_length

    # Create window
    if window == 'hann':
        win = np.hanning(frame_size)
    elif window == 'hamming':
        win = np.hamming(frame_size)
    elif window == 'blackman':
        win = np.blackman(frame_size)
    elif window is None:
        win = np.ones(frame_size)
    else:
        raise ValueError(f"Unsupported window: {window}")

    # Initialize STFT matrix (using rfft for real signals)
    n_freq = frame_size // 2 + 1
    stft_matrix = np.zeros((n_freq, n_frames), dtype=np.complex128)

    # Compute STFT frame by frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_size

        if end <= len(signal):
            frame = signal[start:end] * win
        else:
            # Pad the last frame if needed
            frame = np.pad(signal[start:], (0, end - len(signal))) * win
            
        # Compute FFT
        stft_matrix[:, i] = np.fft.rfft(frame, n=frame_size)

    return stft_matrix


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute magnitude spectrogram from STFT.

    Args:
        stft_matrix: Complex STFT matrix

    Returns:
        Magnitude spectrogram
    """
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """
    Compute power spectrogram from STFT.

    Args:
        stft_matrix: Complex STFT matrix

    Returns:
        Power spectrogram
    """
    return np.abs(stft_matrix) ** 2