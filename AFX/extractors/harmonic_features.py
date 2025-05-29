
"""
Harmonic feature extractors for audio signals (pitch, THD, HNR).
"""
import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor
from AFX.utils.pitch import yin
from scipy.fft import rfft
_EPS = 1e-10

def _pad_frame(frame: np.ndarray, frame_size: int) -> np.ndarray:
    """Pad frame to the required frame size with zeros if needed."""
    if len(frame) < frame_size:
        return np.pad(frame, (0, frame_size - len(frame)))
    return frame

__all__ = [
    'extract_pitch',
    'extract_thd',
    'extract_hnr',
]

@framewise_extractor
def extract_pitch(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 50.0,
    fmax: float = 2000.0,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Estimate pitch (fundamental frequency) using the YIN algorithm.
    
    Implementation based on "YIN, a fundamental frequency estimator for speech and music"
    by Alain de Cheveigné and Hideki Kawahara.
    
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for analysis
        hop_length: Hop length between frames
        fmin: Minimum frequency
        fmax: Maximum frequency
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'pitch' key and np.ndarray of pitch values (Hz)
    Metadata:
        shape: (n_frames,)
        units: Hz
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    try:
        pitch = yin(
            signal, frame_length=frame_size, hop_length=hop_length,
            fmin=fmin, fmax=fmax, sr=sr
        )
    except Exception:
        pitch = np.full((1,), np.nan)
    result = {'pitch': pitch}
    if return_metadata:
        times = np.arange(len(pitch)) * hop_length / sr
        return {'pitch': pitch, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_thd(
    signal: np.ndarray,
    sr: int,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Estimate Total Harmonic Distortion (THD) of the audio signal.

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
    Returns:
        Dict with 'thd' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: ratio
    Note: Fundamental is assumed to be the max FFT bin (excluding DC). This may be inaccurate for noisy signals or signals with strong DC components.
    """
    try:
        yf = np.abs(rfft(signal))
        # Find fundamental (skip DC)
        fundamental_idx = np.argmax(yf[1:]) + 1
        fundamental = yf[fundamental_idx]
        if fundamental == 0:
            return {'thd': np.array([np.nan])}
        harmonics = []
        # Only integer multiples of the fundamental are considered harmonics
        for h in range(2, 6):
            idx = fundamental_idx * h
            if idx < len(yf):
                harmonics.append(yf[idx])
        harmonics = np.array(harmonics)
        thd = np.sqrt(np.sum(harmonics ** 2)) / (fundamental + _EPS)
    except Exception:
        thd = np.nan
    return {'thd': np.array([thd])}

@framewise_extractor
def extract_hnr(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Estimate Harmonics-to-Noise Ratio (HNR) using autocorrelation method.
    
    Uses a custom NumPy-based framing approach to segment the audio signal
    into overlapping frames, then estimates HNR using the autocorrelation method.
    
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for analysis
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'hnr' key and np.ndarray of HNR values
    Metadata:
        shape: (n_frames,)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    n_samples = len(signal)
    n_frames = max(1, 1 + (n_samples - frame_size) // hop_length)
    hnr = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_size, n_samples)
        frame = signal[start:end]
        frame = _pad_frame(frame, frame_size)
        try:
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            if autocorr[0] == 0:
                hnr.append(np.nan)
                continue
            max_autocorr = np.max(autocorr[1:])
            hnr_val = 10 * np.log10((max_autocorr + _EPS) / (autocorr[0] - max_autocorr + _EPS))
            hnr.append(hnr_val)
        except Exception:
            hnr.append(np.nan)
    hnr = np.array(hnr)
    result = {'hnr': hnr}
    if return_metadata:
        times = np.arange(n_frames) * hop_length / sr
        return {'hnr': hnr, 'metadata': {'times': times}}
    return result
