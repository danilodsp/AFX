"""
Constant-Q Transform (CQT) implementation and approximation.
"""
import numpy as np
from typing import Tuple


def cqt_approximation(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = None,
    frame_size: int = 2048
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a CQT approximation using log-spaced frequency bins and STFT.

    This function approximates the Constant-Q Transform by:
    1. Computing STFT with zero-padding for high frequency resolution
    2. Creating logarithmically-spaced frequency bins (geometric series)
    3. Mapping STFT bins to CQT bins using linear interpolation

    Args:
        signal: Input audio signal (1D array)
        sr: Sample rate in Hz
        hop_length: Number of samples between adjacent frames
        n_bins: Total number of CQT frequency bins
        bins_per_octave: Number of bins per octave (typically 12)
        fmin: Minimum frequency in Hz (default: C1 ≈ 32.7 Hz)
        frame_size: STFT frame size for frequency resolution

    Returns:
        Tuple of (cqt_magnitude, frequencies):
        - cqt_magnitude: CQT approximation of shape (n_bins, n_frames)
        - frequencies: Center frequencies of CQT bins in Hz

    Notes:
        - Uses geometric spacing: f[k] = fmin * 2^(k/bins_per_octave)
        - Higher frame_size provides better frequency resolution for low frequencies
        - CQT bins are created by interpolating STFT magnitude spectrum
    """
    from AFX.methods.stft import stft, magnitude_spectrogram

    # Set default minimum frequency (C1 note)
    if fmin is None:
        fmin = 32.70319566257483  # C1 in Hz

    # Compute STFT with high frequency resolution
    stft_matrix = stft(signal, frame_size=frame_size, hop_length=hop_length, 
                       window='hann', center=True)
    magnitude_spec = magnitude_spectrogram(stft_matrix)

    # Create logarithmically-spaced frequency bins (CQT frequencies)
    # f[k] = fmin * 2^(k/bins_per_octave)
    cqt_frequencies = fmin * (2 ** (np.arange(n_bins) / bins_per_octave))

    # Create linear frequency bins for STFT
    stft_frequencies = np.fft.rfftfreq(frame_size, 1/sr)

    # Initialize CQT magnitude array
    n_frames = magnitude_spec.shape[1]
    cqt_magnitude = np.zeros((n_bins, n_frames))

    # Map STFT bins to CQT bins using linear interpolation
    for frame_idx in range(n_frames):
        # Interpolate STFT magnitude spectrum to CQT frequency grid
        cqt_magnitude[:, frame_idx] = np.interp(
            cqt_frequencies, stft_frequencies, magnitude_spec[:, frame_idx]
        )

    return cqt_magnitude, cqt_frequencies


def pseudo_cqt(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = None
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

    Returns:
        CQT magnitude matrix of shape (n_bins, n_frames)
    """
    cqt_mag, _ = cqt_approximation(
        signal, sr, hop_length, n_bins, bins_per_octave, fmin
    )
    return cqt_mag