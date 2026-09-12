"""
Chroma feature computation methods.
"""
import numpy as np


def chroma_from_stft(
    magnitude_spec: np.ndarray,
    sr: int,
    n_fft: int,
    f_ref: float = 261.626,  # C4 frequency
    normalize: bool = True
) -> np.ndarray:
    """
    Compute chroma features from magnitude spectrogram.

    Maps frequency bins from an STFT magnitude spectrogram to 12 chroma bins
    (pitch classes) and aggregates magnitudes across octaves.

    The chroma bin mapping uses the formula:
        chroma_bin = round(12 * log2(f / f_ref)) % 12

    where f_ref is the reference frequency for C4 (default: 261.626 Hz).

    Args:
        magnitude_spec: Magnitude spectrogram of shape (n_freq, n_frames)
        sr: Sample rate in Hz  
        n_fft: FFT size used for the STFT
        f_ref: Reference frequency for C4 in Hz (default: 261.626)
        normalize: Whether to L1 normalize each chroma vector (default: True)

    Returns:
        Chroma features of shape (12, n_frames)

    Notes:
        - Frequency bins are computed as f = bin_index * sr / n_fft
        - DC bin (f=0) is ignored to avoid log(0)
        - Each chroma bin accumulates energy from all corresponding frequency bins
        - Optional L1 normalization makes each chroma vector sum to 1
        - Chroma bins: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    """
    n_freq, n_frames = magnitude_spec.shape

    # Compute frequency for each bin: f = bin_index * sr / n_fft
    freqs = np.arange(n_freq) * sr / n_fft

    # Vectorized mapping of every non-DC bin to its pitch class
    chroma_bins = np.mod(
        np.round(12.0 * np.log2(freqs[1:] / f_ref)).astype(int), 12
    )

    # Fold all frequency bins into their chroma bins (duplicate-safe)
    chroma = np.zeros((12, n_frames))
    np.add.at(chroma, chroma_bins, magnitude_spec[1:, :])

    # Optional L1 normalization per frame
    if normalize:
        # Add small epsilon to avoid division by zero
        norms = np.sum(chroma, axis=0, keepdims=True) + 1e-10
        chroma = chroma / norms

    return chroma


def frequency_to_chroma_bin(freq: float, f_ref: float = 261.626) -> int:
    """
    Map a frequency to its corresponding chroma bin (0-11).

    Uses the formula: chroma_bin = round(12 * log2(f / f_ref)) % 12

    Args:
        freq: Frequency in Hz
        f_ref: Reference frequency for C4 in Hz (default: 261.626)

    Returns:
        Chroma bin index (0-11) where:
        0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
    """
    if freq <= 0:
        raise ValueError(f'frequency must be positive, got {freq}')
    return int(np.round(12 * np.log2(freq / f_ref))) % 12


def extract_chroma_from_cqt(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.70319566257483,
    center: bool = False,
    frame_size: int = 4096,
) -> np.ndarray:
    """
    Extract chroma features from a CQT approximation.

    This function computes a chromagram from a Constant-Q Transform (CQT)
    approximation, capturing harmonic content and mapping it to 12 pitch
    classes. The implementation:

    1. Computes a pseudo-CQT (triangular log-frequency filterbank applied
       to an STFT magnitude spectrogram)
    2. Maps each CQT bin to a chroma bin from its center frequency
       (12 pitch classes, modulo 12)
    3. Aggregates energy into each chroma bin across all octaves

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        hop_length: Hop length between frames in samples (default: 512)
        n_bins: Number of frequency bins in the CQT (default: 84 = 7 octaves * 12)
        bins_per_octave: Number of bins per octave (default: 12)
        fmin: Minimum frequency in Hz (default: 32.70319566257483 = C1)
        center: If True, pad the signal when computing the STFT so that
            frames are centered (mirrors ``librosa`` behaviour).
        frame_size: STFT frame size used for the CQT approximation
            (default 4096 for good low-frequency resolution)

    Returns:
        Chroma features of shape (12, n_frames)

    Notes:
        - CQT frequencies: f[k] = fmin * 2^(k/bins_per_octave)
        - MIDI note mapping: midi = 69 + 12 * log2(freq / 440)
        - Chroma bin: chroma_bin = round(midi) % 12 (0=C, ..., 11=B)
    """
    from AFX.methods.cqt import cqt_approximation

    cqt_magnitude, cqt_frequencies = cqt_approximation(
        signal,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        frame_size=frame_size,
        center=center,
    )
    n_frames = cqt_magnitude.shape[1]

    # Map each CQT bin to its pitch class via its center frequency
    midi_notes = 69.0 + 12.0 * np.log2(cqt_frequencies / 440.0)
    chroma_bins = np.mod(np.round(midi_notes).astype(int), 12)

    # Fold octaves: chroma[c] = sum of all CQT bins in pitch class c
    chroma = np.zeros((12, n_frames))
    np.add.at(chroma, chroma_bins, cqt_magnitude)

    # L1 normalize each chroma vector (make each column sum to 1)
    norms = np.sum(chroma, axis=0, keepdims=True) + 1e-10
    chroma = chroma / norms

    return chroma
