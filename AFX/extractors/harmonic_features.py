
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
    pitch = yin(
        signal, frame_length=frame_size, hop_length=hop_length,
        fmin=fmin, fmax=fmax, sr=sr
    )
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
    Harmonic peaks are searched within +/-2 bins of each integer multiple of
    the fundamental to tolerate spectral leakage.
    """
    yf = np.abs(rfft(signal))
    if len(yf) < 2:
        return {'thd': np.array([np.nan])}
    # Find fundamental (skip DC)
    fundamental_idx = np.argmax(yf[1:]) + 1
    fundamental = yf[fundamental_idx]
    if fundamental == 0:
        return {'thd': np.array([np.nan])}
    harmonics = []
    # Search around integer multiples of the fundamental (leakage-tolerant)
    for h in range(2, 6):
        idx = fundamental_idx * h
        if idx < len(yf):
            lo = max(idx - 2, 0)
            hi = min(idx + 3, len(yf))
            harmonics.append(np.max(yf[lo:hi]))
    harmonics = np.array(harmonics)
    thd = np.sqrt(np.sum(harmonics ** 2)) / (fundamental + _EPS)
    return {'thd': np.array([thd])}

def extract_hnr(
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
    Estimate Harmonics-to-Noise Ratio (HNR) using the autocorrelation method.

    For each frame the normalized autocorrelation r(tau) = R(tau)/R(0) is
    computed, and its maximum is searched only over lags corresponding to
    plausible pitch periods (sr/fmax .. sr/fmin). The peak value r_max
    estimates the fraction of periodic (harmonic) energy, giving:

        HNR = 10 * log10(r_max / (1 - r_max))

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for analysis
        hop_length: Hop length between frames
        fmin: Minimum fundamental frequency considered (Hz)
        fmax: Maximum fundamental frequency considered (Hz)
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

    # Restrict the peak search to plausible pitch lags. Searching all lags
    # would always find autocorr[1] ~ autocorr[0] for any smooth signal,
    # grossly overestimating HNR regardless of harmonicity.
    lag_min = max(1, int(sr / fmax)) if fmax > 0 else 1
    lag_max = min(int(sr / fmin) if fmin > 0 else frame_size - 1, frame_size - 1)

    hnr = []
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_size, n_samples)
        frame = signal[start:end]
        frame = _pad_frame(frame, frame_size)
        autocorr = np.correlate(frame, frame, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]
        if autocorr[0] <= 0 or lag_min >= lag_max:
            hnr.append(np.nan)
            continue
        r = autocorr / autocorr[0]
        r_max = np.max(r[lag_min:lag_max + 1])
        # Clip to the open interval (0, 1) to keep the log-ratio finite
        r_max = np.clip(r_max, _EPS, 1.0 - _EPS)
        hnr_val = 10 * np.log10(r_max / (1.0 - r_max))
        hnr.append(hnr_val)
    hnr = np.array(hnr)
    result = {'hnr': hnr}
    if return_metadata:
        times = np.arange(n_frames) * hop_length / sr
        return {'hnr': hnr, 'metadata': {'times': times}}
    return result
