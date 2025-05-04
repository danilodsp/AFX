"""
Time-domain feature extractors for audio signals.
"""
from typing import Dict, Optional
import numpy as np
from AFX.utils.framewise import framewise_extractor
from scipy.stats import kurtosis

__all__ = [
    'extract_zero_crossing_rate',
    'extract_variance',
    'extract_entropy',
    'extract_crest_factor',
]

def extract_zero_crossing_rate(
    signal: np.ndarray,
    sr: Optional[int] = None,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the zero-crossing rate (ZCR) of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        frame_size: Frame size for ZCR calculation
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'zcr' key and np.ndarray of ZCR values
    Metadata:
        shape: (n_frames,)
        units: float
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    zcr = librosa.feature.zero_crossing_rate(
        signal, frame_length=frame_size, hop_length=hop_length
    )[0]
    result = {'zcr': zcr}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(zcr)), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'zcr': zcr, 'metadata': {'times': times}}
    return result


@framewise_extractor
def extract_variance(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the variance of the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
    Returns:
        Dict with 'variance' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    var = np.var(signal)
    return {'variance': np.array([var])}


@framewise_extractor
def extract_entropy(
    signal: np.ndarray,
    sr: Optional[int] = None,
    num_bins: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the entropy of the audio signal histogram.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        num_bins: Number of bins for histogram
    Returns:
        Dict with 'entropy' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    hist, _ = np.histogram(signal, bins=num_bins, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log2(hist))
    return {'entropy': np.array([entropy])}


@framewise_extractor
def extract_crest_factor(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the crest factor of the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
    Returns:
        Dict with 'crest_factor' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    peak = np.max(np.abs(signal))
    rms = np.sqrt(np.mean(signal ** 2))
    crest = peak / rms if rms > 0 else 0.0
    return {'crest_factor': np.array([crest])}


@framewise_extractor
def extract_kurtosis(
    signal: np.ndarray,
    sr: Optional[int] = None,
    fisher: bool = True,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the kurtosis of the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        fisher: If True, Fisher's definition (normal ==> 0.0)
    Returns:
        Dict with 'kurtosis' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    k = kurtosis(signal, fisher=fisher)
    return {'kurtosis': np.array([k])}


@framewise_extractor
def extract_short_time_energy(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the short-time energy of the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
    Returns:
        Dict with 'short_time_energy' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ste = np.sum(signal ** 2)
    return {'short_time_energy': np.array([ste])}


@framewise_extractor
def extract_energy_ratio(
    signal: np.ndarray,
    sr: Optional[int] = None,
    split: float = 0.5,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the energy ratio between two halves of the signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        split: Fraction to split (default 0.5)
    Returns:
        Dict with 'energy_ratio' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    n = int(len(signal) * split)
    e1 = np.sum(signal[:n] ** 2)
    e2 = np.sum(signal[n:] ** 2)
    ratio = e1 / (e2 + 1e-10)
    return {'energy_ratio': np.array([ratio])}


@framewise_extractor
def extract_sample_entropy(
    signal: np.ndarray,
    sr: Optional[int] = None,
    m: int = 2,
    r: float = 0.2,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the sample entropy of the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        m: Embedding dimension
        r: Tolerance (as a fraction of std)
    Returns:
        Dict with 'sample_entropy' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    def _phi(m):
        N = len(signal)
        x = np.array([signal[i:i + m] for i in range(N - m + 1)])
        C = np.sum(
            np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r * np.std(signal), axis=0
        ) - 1
        return np.sum(C) / ((N - m + 1) * (N - m))
    try:
        se = -np.log(_phi(m + 1) / (_phi(m) + 1e-10))
    except Exception:
        se = np.nan
    return {'sample_entropy': np.array([se])}


@framewise_extractor
def extract_autocorrelation_variance(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the variance of the autocorrelation function of the signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_variance' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    var = np.var(acf)
    return {'autocorr_variance': np.array([var])}


@framewise_extractor
def extract_mobility(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the mobility (RMS of first derivative / RMS of signal).
    Args:
        signal: Audio signal (1D np.ndarray)
    Returns:
        Dict with 'mobility' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    diff = np.diff(signal)
    rms_diff = np.sqrt(np.mean(diff ** 2))
    rms = np.sqrt(np.mean(signal ** 2))
    mobility = rms_diff / (rms + 1e-10)
    return {'mobility': np.array([mobility])}


@framewise_extractor
def extract_complexity(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the complexity (ratio of mobility of first diff to mobility of signal).
    Args:
        signal: Audio signal (1D np.ndarray)
    Returns:
        Dict with 'complexity' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    diff = np.diff(signal)
    diff2 = np.diff(diff)
    rms_diff = np.sqrt(np.mean(diff ** 2))
    rms_diff2 = np.sqrt(np.mean(diff2 ** 2))
    complexity = rms_diff2 / (rms_diff + 1e-10)
    return {'complexity': np.array([complexity])}


@framewise_extractor
def extract_autocorr_snr(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the SNR from the autocorrelation function (peak to mean ratio).
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_snr' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    snr = np.max(acf) / (np.mean(acf) + 1e-10)
    return {'autocorr_snr': np.array([snr])}

@framewise_extractor
def extract_autocorr_zero_crossings(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the number of zero crossings in the autocorrelation function.
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_zero_crossings' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: int
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    zc = np.sum(np.diff(np.sign(acf)) != 0)
    return {'autocorr_zero_crossings': np.array([zc])}

@framewise_extractor
def extract_autocorr_peak_width(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the width of the main peak in the autocorrelation function.
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_peak_width' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: int
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    peak = np.max(acf)
    above_half = np.where(acf >= 0.5 * peak)[0]
    width = above_half[-1] - above_half[0] + 1 if len(above_half) > 1 else 0
    return {'autocorr_peak_width': np.array([width])}

@framewise_extractor
def extract_autocorr_decay_rate(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the decay rate of the autocorrelation envelope.
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_decay_rate' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    try:
        decay = -np.polyfit(
            np.arange(1, len(acf)),
            np.log(np.abs(acf[1:]) + 1e-10),
            1
        )[0]
    except Exception:
        decay = np.nan
    return {'autocorr_decay_rate': np.array([decay])}

@framewise_extractor
def extract_autocorr_coeffs_deviation(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the standard deviation of the autocorrelation coefficients.
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'autocorr_coeffs_deviation' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    std = np.std(acf)
    return {'autocorr_coeffs_deviation': np.array([std])}

@framewise_extractor
def extract_max_peak_autocorr_envelope(
    signal: np.ndarray,
    sr: Optional[int] = None,
    max_lag: int = 100,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the maximum peak of the autocorrelation envelope (excluding zero lag).
    Args:
        signal: Audio signal (1D np.ndarray)
        max_lag: Maximum lag to consider
    Returns:
        Dict with 'max_peak_autocorr_envelope' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    ac = np.correlate(signal, signal, mode='full')
    mid = len(ac) // 2
    acf = ac[mid:mid+max_lag]
    if len(acf) > 1:
        peak = np.max(acf[1:])
    else:
        peak = np.nan
    return {'max_peak_autocorr_envelope': np.array([peak])}
