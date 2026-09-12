"""
Time-domain feature extractors for audio signals.
"""
from typing import Dict, Optional
import numpy as np
from scipy.stats import kurtosis
from AFX.methods.autocorrelation import autocorrelation as _autocorr
from AFX.utils.frame_utils import frame_times
from AFX.utils.framewise import framewise_extractor

__all__ = [
    'extract_zero_crossing_rate',
    'extract_variance',
    'extract_entropy',
    'extract_crest_factor',
    'extract_rms_energy',
    'extract_kurtosis',
    'extract_short_time_energy',
    'extract_energy_ratio',
    'extract_sample_entropy',
    'extract_mobility',
    'extract_complexity',
    'extract_autocorrelation_variance',
    'extract_autocorr_snr',
    'extract_autocorr_zero_crossings',
    'extract_autocorr_peak_width',
    'extract_autocorr_decay_rate',
    'extract_autocorr_coeffs_deviation',
    'extract_max_peak_autocorr_envelope',
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
    # Ensure signal is a numpy array
    signal = np.asarray(signal)

    # Check if signal is shorter than frame_size
    if len(signal) < frame_size:
        # If signal is too short, just compute ZCR for the entire signal
        # and avoid framing
        if len(signal) <= 1:
            # Edge case: can't have zero crossings with 1 sample or less
            zcr = np.array([0.0])
        else:
            # Count sign changes and normalize
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal).astype(int))))
            zcr = np.array([zero_crossings / (len(signal) - 1)])
    else:
        # Create frames of the signal
        frames = []
        for i in range(0, max(1, len(signal) - frame_size + 1), hop_length):
            frames.append(signal[i:i + frame_size])

        # Initialize the output array
        zcr = np.zeros(len(frames))

        # Process each frame
        for i, frame in enumerate(frames):
            # Ensure frame is of expected length
            if len(frame) < frame_size:
                # Pad the last frame if needed
                frame = np.pad(frame, (0, frame_size - len(frame)))
            # Calculate zero-crossing rate
            zero_crossings = np.sum(np.abs(np.diff(np.signbit(frame).astype(int))))
            zcr[i] = zero_crossings / (frame_size - 1)

    result = {'zcr': zcr}
    if return_metadata:
        if sr is None:
            raise ValueError('sr is required when return_metadata=True')
        times = frame_times(len(zcr), sr, hop_length, min(frame_size, len(signal)))
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
    hist, _ = np.histogram(signal, bins=num_bins)
    total = hist.sum()
    if total == 0:
        return {'entropy': np.array([0.0])}
    p = hist[hist > 0] / total
    entropy = -np.sum(p * np.log2(p))
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
def extract_rms_energy(
    signal: np.ndarray,
    sr: Optional[int] = None,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute the RMS energy of the audio signal."""
    rms = np.sqrt(np.mean(signal ** 2))
    return {'rms_energy': np.array([rms])}


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
    max_samples: int = 20000,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the sample entropy of the audio signal.

    Template matching is done with a KD-tree under the Chebyshev metric
    (identical counts to the brute-force pairwise comparison, far faster).
    The worst case is still quadratic, so this feature is intended for
    frame-wise use (pass ``frame_size``/``hop_length``), and signals longer
    than ``max_samples`` are rejected to prevent accidental multi-hour runs
    on whole files.

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate (optional)
        m: Embedding dimension
        r: Tolerance (as a fraction of std)
        max_samples: Maximum allowed signal/frame length
    Returns:
        Dict with 'sample_entropy' key and np.ndarray scalar
    Metadata:
        shape: (1,) or (1, n_frames) if framewise
        units: float
    """
    signal = np.asarray(signal, dtype=np.float64)
    N = len(signal)
    if N > max_samples:
        raise ValueError(
            f'sample_entropy is O(N^2); got {N} samples (max {max_samples}). '
            'Use frame_size/hop_length to compute it frame-wise, or raise '
            'max_samples explicitly.'
        )
    if N < m + 2:
        return {'sample_entropy': np.array([np.nan])}

    tol = r * np.std(signal)

    def _phi(mm):
        from scipy.spatial import cKDTree
        n = N - mm + 1
        if n < 2:
            return 0.0
        x = np.lib.stride_tricks.sliding_window_view(signal, mm)
        tree = cKDTree(x)
        # Ordered pairs (i, j) with Chebyshev distance <= tol, incl. i == j
        pairs = tree.count_neighbors(tree, tol, p=np.inf)
        count = pairs - n  # exclude self-matches
        return count / (n * (n - 1))

    phi_m = _phi(m)
    phi_m1 = _phi(m + 1)
    if phi_m <= 0 or phi_m1 <= 0:
        se = np.nan
    else:
        se = -np.log(phi_m1 / phi_m)
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
    acf = _autocorr(signal, max_lag)
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
    acf = _autocorr(signal, max_lag)
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
    acf = _autocorr(signal, max_lag)
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
    acf = _autocorr(signal, max_lag)
    peak = np.max(acf)
    above_half = np.where(acf >= 0.5 * peak)[0]
    # A single bin above half-peak has width 1 (not 0)
    width = above_half[-1] - above_half[0] + 1 if len(above_half) > 0 else 0
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
    acf = _autocorr(signal, max_lag)
    if len(acf) < 3:
        return {'autocorr_decay_rate': np.array([np.nan])}
    try:
        decay = -np.polyfit(
            np.arange(1, len(acf)),
            np.log(np.abs(acf[1:]) + 1e-10),
            1
        )[0]
    except np.linalg.LinAlgError:
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
    acf = _autocorr(signal, max_lag)
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
    acf = _autocorr(signal, max_lag)
    if len(acf) > 1:
        peak = np.max(acf[1:])
    else:
        peak = np.nan
    return {'max_peak_autocorr_envelope': np.array([peak])}
