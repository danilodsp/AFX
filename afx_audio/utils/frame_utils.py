"""
Utilities for audio frame processing.
"""
import numpy as np

from afx_audio.utils.validation import validate_signal


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


def analysis_frames(
    signal: np.ndarray,
    frame_size: int | None,
    hop_length: int | None,
    center: bool = False,
    pad_mode: str = 'constant',
) -> np.ndarray:
    """
    (n_frames, frame_len) float64 view used by the frame-wise extractors.

    With both ``frame_size`` and ``hop_length`` set: frames of frame_size
    samples every hop_length samples, trailing samples that do not fill a
    frame dropped. ``center`` first pads frame_size // 2 samples on both
    sides (``pad_mode`` as in np.pad), so frame i is centred on sample
    i * hop_length as in librosa. Without ``center``, a signal shorter than
    one frame is analyzed as a single unpadded frame. With either size
    None: the whole signal as one frame.
    """
    signal = validate_signal(signal).astype(np.float64, copy=False)
    if frame_size is None or hop_length is None:
        return signal[np.newaxis, :]
    if center:
        signal = np.pad(signal, frame_size // 2, mode=pad_mode)  # type: ignore[call-overload]
    if len(signal) < frame_size:
        return signal[np.newaxis, :]
    return np.lib.stride_tricks.sliding_window_view(signal, frame_size)[::hop_length]
