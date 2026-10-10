"""
Time-domain feature extractors.

Every extractor takes ``frame_size``, ``hop_length`` and ``center``. With
both sizes set the signal is cut into frames of ``frame_size`` samples
every ``hop_length`` samples (trailing samples that do not fill a frame are
dropped) and the result has shape (n_frames,). ``center=True`` first
zero-pads frame_size // 2 samples on both sides, so frame i is centred on
sample i * hop_length as in librosa and in the centered STFT. With either
size left as None, or an uncentered clip shorter than one frame, the whole
signal is analyzed as a single unpadded frame and the result has shape (1,).

Features are computed for all frames at once on a (n_frames, frame_size)
strided view, in float64.
"""

from typing import Any

import numpy as np

from afx_audio.methods.autocorrelation import autocorrelation as _autocorr
from afx_audio.utils.frame_utils import analysis_frames as _frames
from afx_audio.utils.frame_utils import frame_times
from afx_audio.utils.framewise import framewise_extractor

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

_EPS = 1e-10


def _result(name, values, frames, sr, frame_size, hop_length, center, return_metadata):
    # float64 even for counts, so every feature can mark a frame NaN
    out = {name: np.asarray(values, dtype=np.float64)}
    if return_metadata:
        if sr is None:
            raise ValueError('sr is required when return_metadata=True')
        centered = center and frame_size is not None and hop_length is not None
        out['metadata'] = {
            'times': frame_times(len(frames), sr, hop_length or 0, frames.shape[1], centered)
        }
    return out


def extract_zero_crossing_rate(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = False,
    threshold: float = 1e-10,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Zero-crossing rate: sign changes between consecutive samples divided by the frame length.

    Samples with |x| <= threshold count as zero and zero counts as
    positive, so low-level noise around silence does not register as
    crossings. Equal to librosa.feature.zero_crossing_rate with the same
    center (centered frames are padded by repeating the edge samples, as
    librosa does). Defaults to 2048/512 framing; pass None for one value
    over the whole clip.
    """
    # librosa pads by repeating the edge samples here (zeros elsewhere)
    frames = _frames(signal, frame_size, hop_length, center, pad_mode='edge')
    negative = np.signbit(frames) & (np.abs(frames) > threshold)
    crossings = np.count_nonzero(negative[:, 1:] != negative[:, :-1], axis=1)
    return _result('zcr', crossings / frames.shape[1], frames, sr, frame_size, hop_length, center, return_metadata)


def extract_variance(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Variance of the samples (population, ddof=0)."""
    frames = _frames(signal, frame_size, hop_length, center)
    return _result('variance', frames.var(axis=1), frames, sr, frame_size, hop_length, center, return_metadata)


def _histogram_entropy(frames, num_bins):
    # Same binning as np.histogram(frame, bins=num_bins) for each frame:
    # equal-width bins over [min, max] with max in the last bin, and
    # [min - 0.5, max + 0.5] for a constant frame.
    frames = np.ascontiguousarray(frames)
    lo = frames.min(axis=1, keepdims=True)
    hi = frames.max(axis=1, keepdims=True)
    flat = lo == hi
    lo = np.where(flat, lo - 0.5, lo)
    hi = np.where(flat, hi + 0.5, hi)
    idx = ((frames - lo) * (num_bins / (hi - lo))).astype(np.intp)
    idx = np.clip(idx, 0, num_bins - 1)
    # Correct float rounding at bin edges the way np.histogram does
    edges = np.linspace(lo[:, 0], hi[:, 0], num_bins + 1, axis=1)
    rows = np.arange(len(frames))[:, np.newaxis]
    idx -= frames < edges[rows, idx]
    idx += (frames >= edges[rows, idx + 1]) & (idx != num_bins - 1)

    flat_idx = (idx + rows * num_bins).ravel()
    counts = np.bincount(flat_idx, minlength=len(frames) * num_bins).reshape(len(frames), num_bins)
    p = counts / frames.shape[1]
    with np.errstate(divide='ignore', invalid='ignore'):
        terms = np.where(p > 0, p * np.log2(p), 0.0)
    return -terms.sum(axis=1)


def extract_entropy(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    num_bins: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Shannon entropy (bits) of the amplitude histogram with ``num_bins`` equal-width bins."""
    frames = _frames(signal, frame_size, hop_length, center)
    if frames.shape[1] == 0:
        values = np.zeros(len(frames))
    else:
        values = _histogram_entropy(frames, num_bins)
    return _result('entropy', values, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_crest_factor(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Peak amplitude over RMS. 0 for an all-zero frame."""
    frames = _frames(signal, frame_size, hop_length, center)
    peak = np.abs(frames).max(axis=1)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    crest = np.divide(peak, rms, out=np.zeros_like(rms), where=rms > 0)
    return _result('crest_factor', crest, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_kurtosis(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    fisher: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Sample kurtosis m4 / m2^2 (biased moments), minus 3 when ``fisher``.

    Matches scipy.stats.kurtosis, including NaN for frames whose variance
    is zero to machine precision.
    """
    frames = _frames(signal, frame_size, hop_length, center)
    mean = frames.mean(axis=1, keepdims=True)
    dev2 = (frames - mean) ** 2
    m2 = np.mean(dev2, axis=1)
    m4 = np.mean(dev2 * dev2, axis=1)
    zero = m2 <= (np.finfo(np.float64).eps * mean[:, 0]) ** 2
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.where(zero, np.nan, m4 / m2 ** 2)
    if fisher:
        k = k - 3.0
    return _result('kurtosis', k, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_short_time_energy(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Sum of squared samples."""
    frames = _frames(signal, frame_size, hop_length, center)
    energy = np.sum(frames ** 2, axis=1)
    return _result('short_time_energy', energy, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_rms_energy(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Root mean square of the samples."""
    frames = _frames(signal, frame_size, hop_length, center)
    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    return _result('rms_energy', rms, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_energy_ratio(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    split: float = 0.5,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Energy before ``split`` (fraction of the frame length) over energy after it."""
    frames = _frames(signal, frame_size, hop_length, center)
    n = int(frames.shape[1] * split)
    e1 = np.sum(frames[:, :n] ** 2, axis=1)
    e2 = np.sum(frames[:, n:] ** 2, axis=1)
    return _result('energy_ratio', e1 / (e2 + _EPS), frames, sr, frame_size, hop_length, center, return_metadata)


def _difference_rms(frames):
    """RMS of x, x' and x''; NaN where the frame is too short for the difference."""
    def rms(x):
        if x.shape[1] == 0:
            return np.full(len(x), np.nan)
        return np.sqrt(np.mean(x ** 2, axis=1))

    d1 = np.diff(frames, axis=1)
    return rms(frames), rms(d1), rms(np.diff(d1, axis=1))


def extract_mobility(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Hjorth mobility, RMS(x') / RMS(x) with x' the first difference.

    Hjorth (1970), "EEG analysis based on time domain properties". RMS is
    used instead of standard deviation, so a DC offset lowers the value.
    """
    frames = _frames(signal, frame_size, hop_length, center)
    rms0, rms1, _ = _difference_rms(frames)
    return _result('mobility', rms1 / (rms0 + _EPS), frames, sr, frame_size, hop_length, center, return_metadata)


def extract_complexity(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """RMS(x'') / RMS(x'), second- over first-difference RMS (the mobility of x')."""
    frames = _frames(signal, frame_size, hop_length, center)
    _, rms1, rms2 = _difference_rms(frames)
    return _result('complexity', rms2 / (rms1 + _EPS), frames, sr, frame_size, hop_length, center, return_metadata)


# Above this many candidate pairs the KD-tree count is faster than checking pairs directly
_MAX_CANDIDATE_PAIRS = 120_000


def _sample_entropy_counts_sorted(x, m, tol, order, counts):
    """Unordered matching pairs (B for length m, A for m + 1) among the candidates from the sort."""
    n = len(order)
    p = np.repeat(np.arange(n), counts)
    q = p + 1 + (np.arange(len(p)) - np.repeat(np.cumsum(counts) - counts, counts))
    i, j = order[p], order[q]
    # k = 0 is checked again: the sort window used first[p] + tol, which can
    # round differently from |x[i] - x[j]| <= tol at exact ties
    match = np.ones(len(p), dtype=bool)
    for k in range(m):
        match &= np.abs(x[i + k] - x[j + k]) <= tol
    b = int(match.sum())
    match &= np.abs(x[i + m] - x[j + m]) <= tol
    return b, int(match.sum())


def _sample_entropy_counts_kdtree(x, m, tol, n_templates):
    from scipy.spatial import cKDTree

    def pairs(length):
        templates = np.lib.stride_tricks.sliding_window_view(x, length)[:n_templates]
        tree = cKDTree(templates)
        # count_neighbors counts ordered pairs including i == j
        return (tree.count_neighbors(tree, tol, p=np.inf) - n_templates) // 2

    return pairs(m), pairs(m + 1)


@framewise_extractor
def extract_sample_entropy(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    m: int = 2,
    r: float = 0.2,
    max_samples: int = 20000,
) -> dict[str, Any]:
    """
    Sample entropy -ln(A/B) (Richman and Moorman, 2000, Am. J. Physiol. 278(6)).

    B counts ordered pairs i != j of the first N - m templates of length m
    within Chebyshev distance r * std(signal), A the same for the N - m
    templates of length m + 1, so both counts use the same template set.
    Candidate pairs come from sorting the templates by their first sample;
    when there are too many (long or near-constant frames) a KD-tree counts
    instead. The worst case is still quadratic, so use it frame-wise and
    keep frames below ``max_samples``. NaN when A or B is 0 or the frame
    has fewer than m + 2 samples.
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
    n_templates = N - m
    # Templates sorted by their first sample: the candidates for a match
    # with template p (in that order) are the following ones up to +tol
    order = np.argsort(signal[:n_templates], kind='stable')
    first = signal[:n_templates][order]
    counts = np.searchsorted(first, first + tol, side='right') - np.arange(n_templates) - 1
    if counts.sum() <= _MAX_CANDIDATE_PAIRS:
        b, a = _sample_entropy_counts_sorted(signal, m, tol, order, counts)
    else:
        b, a = _sample_entropy_counts_kdtree(signal, m, tol, n_templates)
    se = -np.log(a / b) if a > 0 and b > 0 else np.nan
    return {'sample_entropy': np.array([se])}


def extract_autocorrelation_variance(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Variance of the raw autocorrelation over lags 0..max_lag-1."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    return _result('autocorr_variance', acf.var(axis=1), frames, sr, frame_size, hop_length, center, return_metadata)


def extract_autocorr_snr(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Peak over mean of the autocorrelation over lags 0..max_lag-1."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    snr = acf.max(axis=1) / (acf.mean(axis=1) + _EPS)
    return _result('autocorr_snr', snr, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_autocorr_zero_crossings(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Number of sign changes of the autocorrelation over lags 0..max_lag-1."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    zc = np.count_nonzero(np.diff(np.sign(acf), axis=1), axis=1)
    return _result('autocorr_zero_crossings', zc, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_autocorr_peak_width(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Lags from the first to the last autocorrelation value at or above half the peak, inclusive."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    above = acf >= 0.5 * acf.max(axis=1, keepdims=True)
    first = np.argmax(above, axis=1)
    last = acf.shape[1] - 1 - np.argmax(above[:, ::-1], axis=1)
    width = np.where(above.any(axis=1), last - first + 1, 0)
    return _result('autocorr_peak_width', width, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_autocorr_decay_rate(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Negative least-squares slope of log|acf| over lags 1..max_lag-1. NaN with fewer than 3 lags."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    if acf.shape[1] < 3:
        decay = np.full(len(frames), np.nan)
    else:
        y = np.log(np.abs(acf[:, 1:]) + _EPS)
        x = np.arange(1, acf.shape[1], dtype=np.float64)
        x -= x.mean()
        decay = -(y @ x) / (x @ x)
    return _result('autocorr_decay_rate', decay, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_autocorr_coeffs_deviation(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Standard deviation of the autocorrelation over lags 0..max_lag-1."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    std = acf.std(axis=1)
    return _result('autocorr_coeffs_deviation', std, frames, sr, frame_size, hop_length, center, return_metadata)


def extract_max_peak_autocorr_envelope(
    signal: np.ndarray,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = False,
    max_lag: int = 100,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Largest autocorrelation value over lags 1..max_lag-1. NaN when only lag 0 exists."""
    frames = _frames(signal, frame_size, hop_length, center)
    acf = _autocorr(frames, max_lag)
    peak = acf[:, 1:].max(axis=1) if acf.shape[1] > 1 else np.full(len(frames), np.nan)
    return _result('max_peak_autocorr_envelope', peak, frames, sr, frame_size, hop_length, center, return_metadata)
