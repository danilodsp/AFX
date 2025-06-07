"""
Utilities for audio frame processing.
"""
import numpy as np
from typing import Tuple


def frame_signal(
    signal: np.ndarray,
    frame_length: int,
    hop_length: int
    ) -> np.ndarray:
    """
    Frame a signal into overlapping frames.

    Args:
        signal: Audio signal (1D np.ndarray)
        frame_length: Length of each frame
        hop_length: Number of samples between adjacent frames

    Returns:
        Framed signal as a 2D array (frame_length, n_frames)
    """
    # Calculate number of frames
    # Ensure at least one frame even if the signal is shorter than frame_length
    n_frames = max(1, 1 + (len(signal) - frame_length) // hop_length)

    # Pad the signal if necessary to get the exact number of frames
    pad_len = (n_frames - 1) * hop_length + frame_length - len(signal)
    if pad_len > 0:
        signal = np.pad(signal, (0, pad_len))

    # Create an output array
    frames = np.zeros((frame_length, n_frames), dtype=signal.dtype)

    # Fill in the frames
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_length
        frames[:, i] = signal[start:end]

    return frames
