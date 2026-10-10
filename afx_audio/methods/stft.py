"""
Short-time Fourier transform with a small result cache.

``stft`` keeps its last six results keyed on a digest of the samples and
the parameters, so extractors that analyze the same signal with the same
framing share one transform.
"""
import hashlib
import threading
from collections import OrderedDict
from functools import lru_cache
from typing import Literal

import numpy as np

from afx_audio.utils.validation import validate_signal

# Result cache: maps (signal digest, dtype, params) -> read-only STFT matrix.
# Keys hold a 32-byte BLAKE2b digest of the samples rather than the samples
# themselves, so the cache costs one STFT matrix per entry, not one signal
# copy as well.
_STFT_CACHE: 'OrderedDict[tuple, np.ndarray]' = OrderedDict()
_STFT_CACHE_MAX_ENTRIES = 6
_STFT_CACHE_LOCK = threading.Lock()
# Skip caching for very large signals to bound memory usage.
_STFT_CACHE_MAX_SIGNAL_BYTES = 32 * 1024 * 1024


def clear_stft_cache() -> None:
    """Clear the memoized STFT results."""
    with _STFT_CACHE_LOCK:
        _STFT_CACHE.clear()


@lru_cache(maxsize=16)
def _get_window(window, frame_size: int) -> np.ndarray:
    """Build (and cache) a periodic analysis window, matching librosa."""
    if window is None:
        win = np.ones(frame_size)
    else:
        from scipy.signal import get_window

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
    pad_mode: Literal['constant', 'reflect', 'edge', 'symmetric'] = 'constant',
) -> np.ndarray:
    """
    Complex STFT, shape (frame_size // 2 + 1, n_frames), as librosa.stft with n_fft = win_length = frame_size.

    ``window`` is any scipy.signal.get_window name, used in its periodic
    form, or None for rectangular. ``center`` pads frame_size // 2 samples
    on both sides with ``pad_mode`` ('constant' as in librosa >= 0.10;
    'reflect' falls back to zeros for signals shorter than the pad). A
    signal shorter than one frame is zero-padded to one frame. float32
    input gives complex64 output (computed in double precision). The
    result is read-only because it is shared through the cache.
    """
    signal = validate_signal(signal)

    use_cache = signal.nbytes <= _STFT_CACHE_MAX_SIGNAL_BYTES
    if use_cache:
        digest = hashlib.blake2b(
            np.ascontiguousarray(signal).view(np.uint8), digest_size=32
        ).digest()
        key = (
            digest, signal.dtype.str, len(signal),
            frame_size, hop_length, window, center, pad_mode,
        )
        with _STFT_CACHE_LOCK:
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

    # Double precision even for float32 input: a single-precision FFT has a
    # noise floor near -140 dB that moment-type features (skewness,
    # kurtosis, bandwidth) weight heavily. The result is stored as complex64.
    stft_matrix = np.fft.rfft(frames * win, n=frame_size, axis=1).T
    if signal.dtype == np.float32:
        stft_matrix = stft_matrix.astype(np.complex64)

    stft_matrix.flags.writeable = False

    if use_cache:
        with _STFT_CACHE_LOCK:
            _STFT_CACHE[key] = stft_matrix
            while len(_STFT_CACHE) > _STFT_CACHE_MAX_ENTRIES:
                _STFT_CACHE.popitem(last=False)

    return stft_matrix


def magnitude_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """|X|."""
    return np.abs(stft_matrix)


def power_spectrogram(stft_matrix: np.ndarray) -> np.ndarray:
    """|X|^2."""
    return np.abs(stft_matrix) ** 2
