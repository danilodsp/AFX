"""
Spectrogram and cepstral extractors: mel spectrogram, MFCC and deltas, GFCC,
CQT and chroma.

All of them start from the shared STFT in AFX.methods.stft (periodic Hann
window, ``frame_size`` = n_fft) and return (n_dims, n_frames) arrays. With
``center=True`` the signal is zero-padded by frame_size // 2 on both sides
first, as in librosa.
"""

from typing import Any

import numpy as np

from AFX.methods.chroma import chroma_from_power, extract_chroma_from_cqt
from AFX.methods.cqt import C1_HZ, cqt_approximation
from AFX.methods.dct import extract_mfcc_coefficients
from AFX.methods.delta import compute_delta
from AFX.methods.gammatone import apply_gammatone_filterbank, gammatone_filterbank
from AFX.methods.mel import apply_mel_filterbank, log_mel_spectrogram, mel_filterbank
from AFX.methods.stft import power_spectrogram, stft
from AFX.utils.cache import cache_mel_filterbank, clear_feature_cache, get_cache_stats, get_cached_mel_filterbank
from AFX.utils.frame_utils import frame_times

__all__ = [
    'extract_mfcc',
    'extract_mfcc_delta',
    'extract_mfcc_delta_delta',
    'extract_chroma_cqt',
    'extract_chroma_stft',
    'extract_cqt',
    'extract_melspectrogram',
    'extract_gfcc',
    'extract_tonnetz',
    'extract_onset_strength',
    'clear_feature_cache',
    'get_cache_stats',
]


def _get_mel_filterbank(n_mels, frame_size, sr, fmin, fmax):
    """Slaney-normalized mel filterbank from the process-wide cache."""
    filterbank = get_cached_mel_filterbank(n_mels, frame_size, sr, fmin, fmax, 'slaney')
    if filterbank is None:
        filterbank = mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm='slaney')
        cache_mel_filterbank(filterbank, n_mels, frame_size, sr, fmin, fmax, 'slaney')
    return filterbank


def _mel_power(signal, sr, frame_size, hop_length, n_mels, fmin, fmax, center):
    power = power_spectrogram(stft(signal, frame_size=frame_size, hop_length=hop_length, window='hann', center=center))
    fmax = sr / 2.0 if fmax is None else fmax
    return apply_mel_filterbank(power, _get_mel_filterbank(n_mels, frame_size, sr, fmin, fmax))


def _compute_mfcc(signal, sr, n_mfcc, frame_size, hop_length, n_mels, fmin, fmax, center, top_db=80.0):
    """STFT power -> mel -> dB (clamped at 1e-10, top_db below the peak) -> orthonormal DCT-II."""
    mel = _mel_power(signal, sr, frame_size, hop_length, n_mels, fmin, fmax, center)
    log_mel = log_mel_spectrogram(mel, amin=1e-10, top_db=top_db)
    return extract_mfcc_coefficients(log_mel, n_mfcc, norm='ortho')


def _with_times(name, values, sr, hop_length, frame_size, center, return_metadata):
    out = {name: values}
    if return_metadata:
        out['metadata'] = {'times': frame_times(values.shape[-1], sr, hop_length, frame_size, center)}
    return out


def extract_mfcc(
    signal: np.ndarray,
    sr: int,
    *,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = False,
    top_db: float | None = 80.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Mel-frequency cepstral coefficients, shape (n_mfcc, n_frames).

    Power spectrogram -> Slaney mel filterbank (n_mels bands, fmin..fmax,
    fmax defaults to sr / 2) -> 10 log10 clamped at 1e-10 and floored
    ``top_db`` below the clip's maximum -> orthonormal DCT-II. This is
    librosa.feature.mfcc with the same n_fft, hop_length and center.
    """
    mfcc = _compute_mfcc(signal, sr, n_mfcc, frame_size, hop_length, n_mels, fmin, fmax, center, top_db)
    return _with_times('mfcc', mfcc, sr, hop_length, frame_size, center, return_metadata)


def _delta(mfcc, width, order):
    # The kernel cannot be wider than the sequence; fewer than 3 frames
    # carry no slope information.
    width = min(width, mfcc.shape[1])
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return np.zeros_like(mfcc)
    return compute_delta(mfcc, width=width, order=order)


def extract_mfcc_delta(
    signal: np.ndarray,
    sr: int,
    *,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = False,
    width: int = 9,
    order: int = 1,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    MFCC deltas, shape (n_mfcc, n_frames), key 'mfcc_delta'.

    Savitzky-Golay derivative over ``width`` frames (clipped to the largest
    odd value not above n_frames) with polynomial fits at the edges, as
    librosa.feature.delta computes it; in the interior, order 1 is the
    regression d[t] = sum_n n x[t+n] / sum_n n^2. ``order`` is the
    derivative order (2 = acceleration). All zeros when fewer than 3 frames
    are available.
    """
    mfcc = _compute_mfcc(signal, sr, n_mfcc, frame_size, hop_length, n_mels, fmin, fmax, center)
    delta = _delta(mfcc, width, order)
    return _with_times('mfcc_delta', delta, sr, hop_length, frame_size, center, return_metadata)


def extract_mfcc_delta_delta(
    signal: np.ndarray,
    sr: int,
    *,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = False,
    width: int = 9,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """Second-order MFCC deltas (acceleration), see extract_mfcc_delta."""
    mfcc = _compute_mfcc(signal, sr, n_mfcc, frame_size, hop_length, n_mels, fmin, fmax, center)
    delta2 = _delta(mfcc, width, 2)
    return _with_times('mfcc_delta_delta', delta2, sr, hop_length, frame_size, center, return_metadata)


def extract_gfcc(
    signal: np.ndarray,
    sr: int,
    *,
    n_gfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_gammatone: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Gammatone frequency cepstral coefficients, shape (n_gfcc, n_frames).

    The MFCC pipeline with the mel filterbank replaced by ``n_gammatone``
    4th-order gammatone magnitude responses on the ERB-rate scale
    (Glasberg and Moore, 1990), each normalized to unit area.
    """
    fmax = sr / 2.0 if fmax is None else fmax
    power = power_spectrogram(stft(signal, frame_size=frame_size, hop_length=hop_length, window='hann', center=center))
    gammatone_spec = apply_gammatone_filterbank(power, gammatone_filterbank(n_gammatone, frame_size, sr, fmin, fmax))
    # Same dB compression as the MFCC path, so the two families share a scale
    log_gammatone = log_mel_spectrogram(gammatone_spec, amin=1e-10, top_db=80.0)
    gfcc = extract_mfcc_coefficients(log_gammatone, n_gfcc, norm='ortho')
    return _with_times('gfcc', gfcc, sr, hop_length, frame_size, center, return_metadata)


def extract_chroma_cqt(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float = C1_HZ,
    center: bool = False,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Chromagram from the STFT-based CQT approximation, shape (12, n_frames).

    Each CQT bin (see extract_cqt) is assigned to pitch class
    round(69 + 12 log2(f / 440)) mod 12 (0 = C) and summed over octaves;
    each frame is L1-normalized.
    """
    chroma = extract_chroma_from_cqt(
        signal, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave,
        fmin=fmin, center=center, frame_size=frame_size,
    )
    return _with_times('chroma_cqt', chroma, sr, hop_length, frame_size, center, return_metadata)


def extract_chroma_stft(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    tuning: float = 0.0,
    normalize: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Chromagram from the STFT power spectrum, shape (12, n_frames), row 0 = C.

    Power spectrogram projected on librosa's chroma filterbank (Gaussian
    pitch-class weights per bin, Gaussian octave weighting centred on 880 Hz),
    each frame scaled to a maximum of 1 when ``normalize``. Identical to
    librosa.feature.chroma_stft(..., tuning=tuning); librosa estimates the
    tuning when it is not given, AFX uses ``tuning`` (fractions of a
    semitone from A440) as is.
    """
    spec = stft(signal, frame_size=frame_size, hop_length=hop_length, window='hann', center=center)
    chroma = chroma_from_power(power_spectrogram(spec), sr=sr, n_fft=frame_size, tuning=tuning, normalize=normalize)
    return _with_times('chroma_stft', chroma, sr, hop_length, frame_size, center, return_metadata)


def extract_cqt(
    signal: np.ndarray,
    sr: int,
    *,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    fmin: float | None = None,
    frame_size: int = 2048,
    center: bool = True,
    n_fft: int | None = None,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Constant-Q magnitude approximated from the STFT, shape (n_bins, n_frames).

    Bin k is centered at fmin * 2^(k / bins_per_octave) (fmin defaults to
    C1, 32.7 Hz) and is a triangular, L1-normalized weighting of the STFT
    magnitudes between its neighbours. This is not a true constant-Q
    transform: the time resolution is that of the ``frame_size`` STFT at
    every frequency, and low bins narrower than the STFT bin spacing fall
    back to the nearest STFT bin. ``n_fft`` is an alias for ``frame_size``.
    """
    if n_fft is not None:
        frame_size = n_fft
    cqt_magnitude, _ = cqt_approximation(
        signal, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave,
        fmin=fmin, frame_size=frame_size, center=center,
    )
    return _with_times('cqt', cqt_magnitude, sr, hop_length, frame_size, center, return_metadata)


def extract_melspectrogram(
    signal: np.ndarray,
    sr: int,
    *,
    n_mels: int = 128,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Mel power spectrogram, shape (n_mels, n_frames).

    |STFT|^2 projected on a Slaney-scale, Slaney-normalized triangular
    filterbank (librosa.feature.melspectrogram with power=2 and htk=False).
    Values are linear power, not dB.
    """
    mel = _mel_power(signal, sr, frame_size, hop_length, n_mels, fmin, fmax, center)
    return _with_times('melspectrogram', mel, sr, hop_length, frame_size, center, return_metadata)


def extract_tonnetz(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    tuning: float = 0.0,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Tonal centroid features (Harte et al., 2006), shape (6, n_frames).

    The L1-normalized chroma_stft is projected on the circles of fifths,
    minor thirds and major thirds (x and y of each), as
    librosa.feature.tonnetz does; librosa defaults to chroma_cqt as input,
    so pass it ``chroma=librosa.feature.chroma_stft(..., tuning=0)`` to
    compare.
    """
    spec = stft(signal, frame_size=frame_size, hop_length=hop_length, window='hann', center=center)
    chroma = chroma_from_power(power_spectrogram(spec), sr=sr, n_fft=frame_size, tuning=tuning, normalize=False)
    total = chroma.sum(axis=0, keepdims=True)
    chroma = chroma / np.where(total < np.finfo(chroma.dtype).tiny, 1.0, total)
    scale = np.array([7.0 / 6, 7.0 / 6, 3.0 / 2, 3.0 / 2, 2.0 / 3, 2.0 / 3])
    angles = np.multiply.outer(scale, np.arange(12, dtype=np.float64))
    angles[::2] -= 0.5  # even rows are the sine components
    radius = np.array([1, 1, 1, 1, 0.5, 0.5])[:, np.newaxis]
    tonnetz = (radius * np.cos(np.pi * angles)) @ chroma
    return _with_times('tonnetz', tonnetz, sr, hop_length, frame_size, center, return_metadata)


def extract_onset_strength(
    signal: np.ndarray,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    lag: int = 1,
    max_size: int = 1,
    return_metadata: bool = False,
) -> dict[str, Any]:
    """
    Spectral-flux onset strength, shape (n_frames,), as librosa.onset.onset_strength.

    The centered mel power spectrogram in dB (ref 1, top_db 80) is
    differenced over ``lag`` frames against a ``max_size``-bin maximum
    filter of itself along frequency, negative changes are dropped and the
    bands averaged. The envelope is shifted by lag + frame_size // (2 *
    hop_length) frames so peaks line up with onsets in time, and trimmed
    to the number of mel frames.
    """
    if lag < 1 or max_size < 1:
        raise ValueError('lag and max_size must be positive')
    mel = _mel_power(signal, sr, frame_size, hop_length, n_mels, fmin, fmax, True)
    db = 10.0 * np.log10(np.maximum(mel, 1e-10))
    db = np.maximum(db, db.max() - 80.0)
    if max_size == 1:
        ref = db
    else:
        from scipy.ndimage import maximum_filter1d

        ref = maximum_filter1d(db, max_size, axis=0)
    onset = np.maximum(0.0, db[:, lag:] - ref[:, :-lag]).mean(axis=0)
    onset = np.pad(onset, (lag + frame_size // (2 * hop_length), 0))[:db.shape[1]]
    return _with_times('onset_strength', onset, sr, hop_length, frame_size, True, return_metadata)
