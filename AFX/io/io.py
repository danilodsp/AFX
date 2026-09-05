"""
Audio I/O utilities for loading audio files.
"""
import math
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import scipy.signal


def load_audio(path: str, sr: Optional[int] = 22050) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 signal and sample rate.

    Args:
        path: Path to audio file
        sr: Target sample rate. If None, the file's native sample rate is
            kept and returned unchanged.
    Returns:
        Tuple of (signal, sample_rate)
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file contains no audio samples.
        RuntimeError: If the file cannot be read for any other reason.
    """
    try:
        signal, native_sr = sf.read(path, always_2d=True, dtype='float32')
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f'Error loading audio file {path}: {e}') from e

    if signal.shape[0] == 0:
        raise ValueError(f'Audio file is empty: {path}')

    # Downmix to mono via mean over channels
    if signal.shape[1] > 1:
        signal = np.mean(signal, axis=1)
    else:
        signal = signal[:, 0]

    if sr is None or sr == native_sr:
        return signal.astype(np.float32), int(native_sr)

    # Polyphase resampling: reduce the up/down ratio by the GCD
    g = math.gcd(int(sr), int(native_sr))
    up = int(sr) // g
    down = int(native_sr) // g
    signal = scipy.signal.resample_poly(signal, up, down)
    return signal.astype(np.float32), int(sr)
