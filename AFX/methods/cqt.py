"""
Constant-Q Transform (CQT) implementation and approximation.
"""
import numpy as np
from typing import Optional, Tuple

C1_HZ = 32.70319566257483  # C1 in Hz


def cqt_filterbank(
    n_bins: int,
    bins_per_octave: int,
    fmin: float,
    sr: int,
    frame_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a triangular log-frequency filterbank projecting STFT bins onto
    geometrically spaced CQT bins.

    Each CQT bin k has a triangular filter centered at
    f[k] = fmin * 2^(k / bins_per_octave), spanning from the previous to the
    next bin center. Filters are L1-normalized so each output bin aggregates
    (rather than point-samples) the STFT energy in its band. If a filter is
    narrower than the STFT bin spacing (possible at very low frequencies),
    it falls back to the nearest STFT bin so no output row is all-zero.

    Args:
        n_bins: Number of CQT bins
        bins_per_octave: Bins per octave (typically 12)
        fmin: Minimum (first bin) frequency in Hz
        sr: Sample rate in Hz
        frame_size: STFT frame size (n_fft)

    Returns:
        Tuple of (filterbank, frequencies):
        - filterbank: shape (n_bins, frame_size // 2 + 1)
        - frequencies: CQT bin center frequencies in Hz, shape (n_bins,)

    Raises:
        ValueError: If the highest CQT bin exceeds the Nyquist frequency.
    """
    cqt_frequencies = fmin * (2.0 ** (np.arange(n_bins) / bins_per_octave))
    nyquist = sr / 2.0
    if cqt_frequencies[-1] > nyquist:
        raise ValueError(
            f'Highest CQT bin ({cqt_frequencies[-1]:.1f} Hz) exceeds Nyquist '
            f'({nyquist:.1f} Hz). Reduce n_bins or fmin, or raise sr.'
        )

    stft_frequencies = np.fft.rfftfreq(frame_size, 1.0 / sr)

    # Triangle edges: previous / next bin centers (extrapolated at the ends)
    ratio = 2.0 ** (1.0 / bins_per_octave)
    left = cqt_frequencies / ratio
    right = cqt_frequencies * ratio

    rising = (stft_frequencies[None, :] - left[:, None]) / (
        cqt_frequencies[:, None] - left[:, None]
    )
    falling = (right[:, None] - stft_frequencies[None, :]) / (
        right[:, None] - cqt_frequencies[:, None]
    )
    filterbank = np.maximum(0.0, np.minimum(rising, falling))

    # Fallback for filters too narrow to cover any STFT bin
    empty = filterbank.sum(axis=1) == 0
    if np.any(empty):
        nearest = np.argmin(
            np.abs(stft_frequencies[None, :] - cqt_frequencies[empty, None]),
            axis=1,
        )
        filterbank[np.flatnonzero(empty), nearest] = 1.0

    # L1-normalize each filter so output magnitude is a weighted average
    filterbank /= filterbank.sum(axis=1, keepdims=True)

    return filterbank, cqt_frequencies


def cqt_approximation(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: Optional[float] = None,
    frame_size: int = 2048,
    center: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a CQT approximation (pseudo-CQT) from an STFT.

    This function approximates the Constant-Q Transform by:
    1. Computing an STFT
    2. Projecting the magnitude spectrum onto geometrically spaced bins
       through a triangular log-frequency filterbank (energy aggregation,
       not point-sampling)

    Args:
        signal: Input audio signal (1D array)
        sr: Sample rate in Hz
        hop_length: Number of samples between adjacent frames
        n_bins: Total number of CQT frequency bins
        bins_per_octave: Number of bins per octave (typically 12)
        fmin: Minimum frequency in Hz (default: C1 ≈ 32.7 Hz)
        frame_size: STFT frame size for frequency resolution
        center: If True, pad the signal so frames are centered

    Returns:
        Tuple of (cqt_magnitude, frequencies):
        - cqt_magnitude: CQT approximation of shape (n_bins, n_frames)
        - frequencies: Center frequencies of CQT bins in Hz

    Notes:
        - Uses geometric spacing: f[k] = fmin * 2^(k/bins_per_octave)
        - Higher frame_size provides better resolution at low frequencies
        - Raises ValueError if the top bin exceeds Nyquist
    """
    from AFX.methods.stft import stft, magnitude_spectrogram

    # Set default minimum frequency (C1 note)
    if fmin is None:
        fmin = C1_HZ

    stft_matrix = stft(signal, frame_size=frame_size, hop_length=hop_length,
                       window='hann', center=center)
    magnitude_spec = magnitude_spectrogram(stft_matrix)

    filterbank, cqt_frequencies = cqt_filterbank(
        n_bins, bins_per_octave, fmin, sr, frame_size
    )
    cqt_magnitude = filterbank @ magnitude_spec

    return cqt_magnitude, cqt_frequencies


def pseudo_cqt(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: Optional[float] = None,
    frame_size: int = 2048,
) -> np.ndarray:
    """
    Simplified CQT approximation for chroma feature extraction.

    This is a streamlined version of cqt_approximation() that returns
    only the magnitude matrix, optimized for chroma computation.

    Args:
        signal: Input audio signal (1D array)
        sr: Sample rate in Hz
        hop_length: Number of samples between adjacent frames
        n_bins: Total number of CQT frequency bins
        bins_per_octave: Number of bins per octave
        fmin: Minimum frequency in Hz (default: C1)
        frame_size: STFT frame size for frequency resolution

    Returns:
        CQT magnitude matrix of shape (n_bins, n_frames)
    """
    cqt_mag, _ = cqt_approximation(
        signal, sr, hop_length, n_bins, bins_per_octave, fmin, frame_size
    )
    return cqt_mag
