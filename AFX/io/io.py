"""
Audio I/O utilities for loading audio files.
"""
from typing import Tuple

import numpy as np
import soundfile as sf
import scipy.signal

def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 signal and sample rate.
    Args:
        path: Path to audio file
        sr: Target sample rate
    Returns:
        Tuple of (signal, sample_rate)
    """
    try:
        signal, sample_rate = sf.read(path, always_2d=True)
        # Convert to mono if needed
        if signal.shape[1] > 1:
            signal = np.mean(signal, axis=1)
        else:
            signal = signal[:, 0]
        # Resample if needed
        if sample_rate != sr:
            num_samples = int(len(signal) * sr / sample_rate)
            signal = scipy.signal.resample(signal, num_samples)
            sample_rate = sr
        return signal.astype(np.float32), sample_rate
    except Exception as e:
        raise RuntimeError(f'Error loading audio file {path}: {e}')
