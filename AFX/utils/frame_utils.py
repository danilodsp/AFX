"""
Utilities for audio frame processing.
"""
import numpy as np


def frame_times(
    n_frames: int,
    sr: float,
    hop_length: int,
    frame_size: int,
    center: bool = False,
) -> np.ndarray:
    """
    Time (seconds) of the center of each analysis frame.

    With ``center=False`` frame i covers samples [i*hop, i*hop + frame_size)
    of the original signal, so its center is at i*hop + frame_size//2.
    With ``center=True`` the signal was padded by frame_size//2 first and
    frame i is centered on sample i*hop. Either way the result is
    ``librosa.frames_to_time(i, sr, hop_length, n_fft=None if center else
    frame_size)``, so features computed with different ``center`` settings
    line up on the same axis.
    """
    offset = 0 if center else frame_size // 2
    return (np.arange(n_frames) * hop_length + offset) / float(sr)


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
        Framed signal as a 2D array (frame_length, n_frames). A signal
        shorter than frame_length is zero-padded to one frame. Trailing
        samples that do not fill a whole frame are dropped, matching the
        STFT framing (n_frames = 1 + (n - frame_length) // hop_length).
    """
    signal = np.asarray(signal)
    if len(signal) < frame_length:
        signal = np.pad(signal, (0, frame_length - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)
    return np.ascontiguousarray(frames[::hop_length].T)
