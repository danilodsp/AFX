"""
Audio I/O utilities for loading audio files.
"""
from typing import Tuple
import numpy as np

def load_audio(path: str, sr: int = 22050) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and return a mono float32 signal and sample rate.
    Args:
        path: Path to audio file
        sr: Target sample rate
    Returns:
        Tuple of (signal, sample_rate)
    """
    import librosa
    try:
        signal, sample_rate = librosa.load(path, sr=sr, mono=True)
        return signal.astype(np.float32), sample_rate
    except Exception as e:
        raise RuntimeError(f'Error loading audio file {path}: {e}')
