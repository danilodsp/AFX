"""
Harmonic feature extractors for audio signals (pitch, THD, HNR).
"""
import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor

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
    Estimate pitch (fundamental frequency) using librosa's YIN algorithm.
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
    import librosa
    pitch = librosa.yin(
        signal, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_size, hop_length=hop_length
    )
    result = {'pitch': pitch}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(pitch)), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
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
    """
    from scipy.fft import rfft
    yf = np.abs(rfft(signal))
    fundamental_idx = np.argmax(yf)
    fundamental = yf[fundamental_idx]
    harmonics = yf[2*fundamental_idx:]
    thd = np.sqrt(np.sum(harmonics ** 2)) / fundamental \
        if fundamental > 0 else 0.0
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
    import librosa
    frames = librosa.util.frame(
        signal, frame_length=frame_size, hop_length=hop_length
    )
    hnr = []
    for frame in frames.T:
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] == 0:
            hnr.append(0.0)
            continue
        max_autocorr = np.max(autocorr[1:])
        hnr_val = 10 * np.log10(
            max_autocorr / (autocorr[0] - max_autocorr + 1e-10)
        )
        hnr.append(hnr_val)
    hnr = np.array(hnr)
    result = {'hnr': hnr}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(hnr)), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'hnr': hnr, 'metadata': {'times': times}}
    return result
