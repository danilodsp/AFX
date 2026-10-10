"""
Chroma: librosa's filterbank on power spectra, plus the hard-assignment folds used by chroma_cqt.
"""
import numpy as np


def _normalize_columns(x: np.ndarray, norm: float) -> np.ndarray:
    """Divide each column by its L-norm; columns with norm below the dtype's tiny stay unchanged (librosa)."""
    if norm == np.inf:
        length = np.max(np.abs(x), axis=0, keepdims=True)
    else:
        length = np.sum(np.abs(x) ** norm, axis=0, keepdims=True) ** (1.0 / norm)
    length = np.where(length < np.finfo(x.dtype).tiny, 1.0, length)
    return x / length


def chroma_filterbank(
    sr: int,
    n_fft: int,
    n_chroma: int = 12,
    tuning: float = 0.0,
    ctroct: float = 5.0,
    octwidth: float | None = 2.0,
) -> np.ndarray:
    """
    STFT-to-chroma weights, shape (n_chroma, n_fft // 2 + 1), equal to librosa.filters.chroma.

    Every FFT bin spreads over neighbouring pitch classes with a Gaussian
    whose width is the bin's spacing in chroma units (so low bins are
    broad), each bin's weights are L2-normalized, then scaled by a Gaussian
    over octaves centred ``ctroct`` octaves above 27.5 Hz (5 -> 880 Hz) with
    ``octwidth`` octaves of standard deviation. Row 0 is C. ``tuning`` is
    the deviation from A440 in fractions of a chroma bin.
    """
    frequencies = np.linspace(0, sr, n_fft, endpoint=False)[1:]
    a440 = 440.0 * 2.0 ** (tuning / n_chroma)
    frqbins = n_chroma * np.log2(frequencies / (a440 / 16))
    # The 0 Hz bin gets a made-up position 1.5 octaves below bin 1, so it is
    # broad and spread evenly
    frqbins = np.concatenate(([frqbins[0] - 1.5 * n_chroma], frqbins))
    binwidthbins = np.concatenate((np.maximum(frqbins[1:] - frqbins[:-1], 1.0), [1]))

    half = np.round(n_chroma / 2.0)
    dist = np.subtract.outer(frqbins, np.arange(n_chroma, dtype=np.float64)).T
    dist = np.remainder(dist + half + 10 * n_chroma, n_chroma) - half
    wts = _normalize_columns(np.exp(-0.5 * (2 * dist / binwidthbins[np.newaxis, :]) ** 2), 2)
    if octwidth is not None:
        wts *= np.exp(-0.5 * (((frqbins / n_chroma - ctroct) / octwidth) ** 2))[np.newaxis, :]
    wts = np.roll(wts, -3 * (n_chroma // 12), axis=0)
    return np.ascontiguousarray(wts[:, :1 + n_fft // 2])


def chroma_from_power(
    power_spec: np.ndarray,
    sr: int,
    n_fft: int,
    tuning: float = 0.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Chromagram (12, n_frames) from a power spectrogram (n_fft // 2 + 1, n_frames).

    Projection through chroma_filterbank, then each frame divided by its
    maximum when ``normalize`` (frames that are all zero stay zero). Same
    as librosa.feature.chroma_stft with the given tuning.
    """
    chroma = chroma_filterbank(sr, n_fft, tuning=tuning) @ power_spec
    return _normalize_columns(chroma, np.inf) if normalize else chroma


def chroma_from_stft(
    magnitude_spec: np.ndarray,
    sr: int,
    n_fft: int,
    f_ref: float = 261.626,  # C4 frequency
    normalize: bool = True
) -> np.ndarray:
    """
    Hard-assignment chromagram (12, n_frames) from a magnitude spectrogram.

    Every non-DC bin adds its magnitude to pitch class
    round(12 log2(f / f_ref)) mod 12, with f_ref = C4 so class 0 is C; with
    ``normalize`` each frame sums to 1. extract_chroma_stft uses
    chroma_from_power instead.
    """
    n_freq, n_frames = magnitude_spec.shape

    # Compute frequency for each bin: f = bin_index * sr / n_fft
    freqs = np.arange(n_freq) * sr / n_fft

    # Vectorized mapping of every non-DC bin to its pitch class
    chroma_bins = np.mod(
        np.round(12.0 * np.log2(freqs[1:] / f_ref)).astype(int), 12
    )

    # Fold all frequency bins into their chroma bins with a 0/1 projection
    projection = np.zeros((12, n_freq - 1))
    projection[chroma_bins, np.arange(n_freq - 1)] = 1.0
    chroma = projection @ magnitude_spec[1:, :]

    if normalize:
        # The epsilon leaves a silent frame at 0 instead of 0 / 0
        norms = np.sum(chroma, axis=0, keepdims=True) + 1e-10
        chroma = chroma / norms

    return chroma


def frequency_to_chroma_bin(freq: float, f_ref: float = 261.626) -> int:
    """Pitch class 0..11 (0 = C) of ``freq``: round(12 log2(freq / f_ref)) mod 12."""
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
    Chromagram (12, n_frames) from the STFT-based CQT approximation.

    Each CQT bin goes to pitch class round(69 + 12 log2(f / 440)) mod 12 of
    its center frequency, classes are summed over octaves and each frame is
    L1-normalized (sum 1, plus 1e-10 against division by zero).
    """
    from afx_audio.methods.cqt import cqt_approximation

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

    # Map each CQT bin to its pitch class via its center frequency
    midi_notes = 69.0 + 12.0 * np.log2(cqt_frequencies / 440.0)
    chroma_bins = np.mod(np.round(midi_notes).astype(int), 12)

    # Fold octaves: chroma[c] = sum of all CQT bins in pitch class c
    projection = np.zeros((12, len(chroma_bins)))
    projection[chroma_bins, np.arange(len(chroma_bins))] = 1.0
    chroma = projection @ cqt_magnitude

    norms = np.sum(chroma, axis=0, keepdims=True) + 1e-10
    chroma = chroma / norms

    return chroma
