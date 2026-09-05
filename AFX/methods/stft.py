"""
Short-Time Fourier Transform (STFT) implementation.

The ``stft`` function memoizes its most recent results (keyed on the signal
content and parameters), so multiple extractors analyzing the same signal
with the same settings share a single STFT computation instead of each
recomputing it.
"""
from collections import OrderedDict
from functools import lru_cache

import numpy as np
from scipy.signal import get_window

# Result cache: maps (signal bytes, dtype, params) -> read-only STFT matrix.
_STFT_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_STFT_CACHE_MAX_ENTRIES = 6
# Skip caching for very large signals to bound memory usage.
_STFT_CACHE_MAX_SIGNAL_BYTES = 32 * 1024 * 1024


def clear_stft_cache() -> None:
    """Clear the memoized STFT results."""
    _STFT_CACHE.clear()


@lru_cache(maxsize=16)
def _get_window(window, frame_size: int) -> np.ndarray:
    """Build (and cache) a periodic analysis window, matching librosa."""
    if window is None:
        win = np.ones(frame_size)
    else:
        # fftbins=True gives the periodic variant used for spectral analysis
        win = get_window(window, frame_size, fftbins=True)
    win.flags.writeable = False
    return win


def stft(
    signal: np.ndarray,
    frame_size: int = 2048,
    hop_length: int = 512,
    window: str = 'hann',
    center: bool = True,
    pad_mode: str = 'constant',
) -> np.ndarray:
    """
    Compute the Short-Time Fourier Transform (STFT) of a signal.

    Args:
        signal: Input signal (1D array)
        frame_size: Length of each frame (n_fft)
        hop_length: Number of samples between adjacent frames
        window: Window function name accepted by ``scipy.signal.get_window``
            ('hann', 'hamming', 'blackman', ...) or None for rectangular
        center: Whether to center frames around the signal
        pad_mode: np.pad mode used for center padding (default 'constant',
            matching librosa >= 0.10)

    Returns:
        Complex STFT matrix of shape (n_freq, n_frames) where
        n_freq = frame_size // 2 + 1. The returned array is read-only
        (results are shared through an internal cache).
    """
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(f'stft expects a 1-D signal, got shape {signal.shape}')
    if len(signal) == 0:
        raise ValueError('stft received an empty signal')

    use_cache = signal.nbytes <= _STFT_CACHE_MAX_SIGNAL_BYTES
    if use_cache:
        key = (
            signal.tobytes(), signal.dtype.str,
            frame_size, hop_length, window, center, pad_mode,
        )
        cached = _STFT_CACHE.get(key)
        if cached is not None:
            _STFT_CACHE.move_to_end(key)
            return cached

    # Center padding if requested
    if center:
        pad_width = frame_size // 2
        if pad_mode == 'reflect' and len(signal) <= pad_width:
            # reflect cannot pad wider than the signal; fall back to zeros
            signal = np.pad(signal, (pad_width, pad_width), mode='constant')
        else:
            signal = np.pad(signal, (pad_width, pad_width), mode=pad_mode)

    # Pad the signal if it's shorter than one frame
    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)), mode='constant')

    win = _get_window(window, frame_size)

    # Vectorized framing: (n_frames, frame_size) view, then one batched rfft
    frames = np.lib.stride_tricks.sliding_window_view(
        signal, frame_size
    )[::hop_length]

    # Preserve float32 precision as complex64 to halve memory
    fft_input = frames * win
    if signal.dtype == np.float32:
        stft_matrix = np.fft.rfft(fft_input, n=frame_size, axis=1
                                  ).astype(np.complex64).T
    else:
        stft_matrix = np.fft.rfft(fft_input, n=frame_size, axis=1).T

    stft_matrix.flags.writeable = False

    if use_cache:
        _STFT_CACHE[key] = stft_matrix
        while len(_STFT_CACHE) > _STFT_CACHE_MAX_ENTRIES:
            _STFT_CACHE.popitem(last=False)

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
