"""
Chroma feature computation methods.
"""
import numpy as np
from typing import Optional


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

    # Initialize chroma matrix
    chroma = np.zeros((12, n_frames))

    # Skip DC bin (freq=0) to avoid log(0)
    for bin_idx in range(1, n_freq):
        freq = freqs[bin_idx]

        # Map frequency to chroma bin using the formula:
        # chroma_bin = round(12 * log2(f / f_ref)) % 12
        chroma_bin = int(np.round(12 * np.log2(freq / f_ref))) % 12

        # Accumulate magnitude for this chroma bin across all frames
        chroma[chroma_bin, :] += magnitude_spec[bin_idx, :]

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
        return 0  # Handle edge case
    return int(np.round(12 * np.log2(freq / f_ref))) % 12


def extract_chroma_from_cqt(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = 32.70319566257483
) -> np.ndarray:
    """
    Extract chroma features from a CQT approximation.

    This function computes a chromagram from a Constant-Q Transform (CQT) approximation,
    capturing harmonic content and mapping it to 12 pitch classes. The implementation:

    1. Computes a CQT approximation using log-spaced frequency bins and STFT
    2. Maps frequency bins to chroma bins (12 pitch classes, modulo 12)
    3. Aggregates energy into each chroma bin across all octaves

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        hop_length: Hop length between frames in samples (default: 512)
        n_bins: Number of frequency bins in the CQT (default: 84 = 7 octaves * 12)
        bins_per_octave: Number of bins per octave (default: 12)
        fmin: Minimum frequency in Hz (default: 32.70319566257483 = C1)

    Returns:
        Chroma features of shape (12, n_frames)

    Notes:
        - Uses geometric frequency spacing to approximate CQT behavior
        - CQT frequencies: f[k] = fmin * 2^(k/bins_per_octave)
        - MIDI note mapping: midi = 69 + 12 * log2(freq / 440)
        - Chroma bin: chroma_bin = midi % 12
    """
    from AFX.methods.stft import stft

    # Compute STFT with appropriate frame size for good frequency resolution
    frame_size = 4096  # Use larger frame size for better frequency resolution at low frequencies
    magnitude_spec = np.abs(stft(signal, frame_size=frame_size, hop_length=hop_length))
    n_freq, n_frames = magnitude_spec.shape

    # Compute frequency for each STFT bin
    freqs = np.arange(n_freq) * sr / frame_size

    # Initialize chroma matrix
    chroma = np.zeros((12, n_frames))

    # For each STFT frequency bin, map to the appropriate chroma bin
    for bin_idx in range(1, n_freq):  # Skip DC bin
        freq = freqs[bin_idx]

        # Skip frequencies below our range
        if freq < fmin:
            continue
            
        # Map frequency to MIDI note number
        # MIDI note 69 = A4 = 440 Hz
        # MIDI note = 69 + 12 * log2(freq / 440)
        midi_note = 69 + 12 * np.log2(freq / 440.0)

        # Map to chroma bin (0-11, where 0=C, 1=C#, ..., 11=B)
        # MIDI note 60 = C4, so chroma_bin = (midi_note - 60) % 12
        # But we can also use: chroma_bin = midi_note % 12
        chroma_bin = int(np.round(midi_note)) % 12

        # Accumulate magnitude for this chroma bin
        chroma[chroma_bin, :] += magnitude_spec[bin_idx, :]

    # L1 normalize each chroma vector (make each column sum to 1)
    norms = np.sum(chroma, axis=0, keepdims=True) + 1e-10
    chroma = chroma / norms

    return chroma
