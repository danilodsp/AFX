"""
Differentiable, GPU-capable feature transforms implemented in pure PyTorch.

These functions mirror the NumPy implementations in ``afx_audio.methods`` and
``afx_audio.extractors`` (same windowing, padding, filterbank and normalization
conventions), so numerical results match the CPU/NumPy path to floating
point tolerance. All functions:

- accept a 1-D ``(time,)`` or batched 2-D ``(batch, time)`` waveform tensor,
- run on whatever device/dtype the input tensor lives on (CPU or CUDA),
- are composed of differentiable ops, so gradients flow back to the
  waveform (except where noted, e.g. ``zero_crossing_rate`` and
  ``spectral_rolloff`` which involve inherently non-differentiable
  counting/selection).

Constant tensors (windows, mel filterbanks, DCT matrices) can be
precomputed once and passed in; ``afx_audio.torch.AFXFeatureModule`` does this
and registers them as buffers so they move with ``.to(device)``.
"""

import numpy as np
import torch

from afx_audio.methods.chroma import chroma_filterbank as _np_chroma_filterbank
from afx_audio.methods.gammatone import gammatone_filterbank as _np_gammatone_filterbank
from afx_audio.methods.mel import mel_filterbank as _np_mel_filterbank

_EPS = 1e-10

__all__ = [
    'stft',
    'magnitude_spectrogram',
    'power_spectrogram',
    'hann_window',
    'mel_filterbank',
    'dct_matrix',
    'melspectrogram',
    'log_mel_spectrogram',
    'mfcc',
    'spectral_centroid',
    'spectral_bandwidth',
    'spectral_rolloff',
    'spectral_entropy',
    'spectral_flatness',
    'spectral_flux',
    'zero_crossing_rate',
    'rms_energy',
    'spectral_skewness',
    'spectral_slope',
    'spectral_kurtosis',
    'spectral_crest',
    'spectral_decrease',
    'spectral_contrast',
    'chroma_filterbank',
    'chroma_stft',
    'delta',
    'gfcc',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    # sqrt has an infinite derivative at 0, which turns into NaN gradients for
    # silent frames; route zeros through a dummy value and return 0 there.
    positive = x > 0
    return torch.where(positive, torch.sqrt(torch.where(positive, x, torch.ones_like(x))), torch.zeros_like(x))


def _as_batch(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """Promote (time,) to (1, time). Returns (tensor, was_unbatched)."""
    if not isinstance(x, torch.Tensor):
        raise TypeError(f'expected a torch.Tensor waveform, got {type(x).__name__}; use torch.as_tensor(x)')
    if not x.is_floating_point():
        raise TypeError(f'waveform must be a floating-point tensor, got {x.dtype}; scale integer PCM first')
    if x.dim() == 1:
        return x.unsqueeze(0), True
    if x.dim() == 2:
        return x, False
    raise ValueError(
        f'Expected 1-D (time,) or 2-D (batch, time) waveform, '
        f'got shape {tuple(x.shape)}'
    )


def _unbatch(x: torch.Tensor, was_unbatched: bool) -> torch.Tensor:
    return x.squeeze(0) if was_unbatched else x


def hann_window(frame_size: int, *, device=None, dtype=None) -> torch.Tensor:
    """Periodic Hann window, matching scipy.signal.get_window('hann', n,
    fftbins=True) used by the NumPy STFT path."""
    return torch.hann_window(
        frame_size, periodic=True, device=device, dtype=dtype
    )


def _rfft_freqs(frame_size: int, sr: int, *, device, dtype) -> torch.Tensor:
    """Frequencies of the rfft bins (0 .. sr/2 inclusive)."""
    return torch.fft.rfftfreq(frame_size, d=1.0 / sr).to(
        device=device, dtype=dtype
    )


# ---------------------------------------------------------------------------
# Core transforms
# ---------------------------------------------------------------------------

def stft(
    signal: torch.Tensor,
    frame_size: int = 2048,
    hop_length: int = 512,
    window: torch.Tensor | None = None,
    center: bool = True,
    pad_mode: str = 'constant',
) -> torch.Tensor:
    """
    Short-Time Fourier Transform, mirroring ``afx_audio.methods.stft.stft``.

    Args:
        signal: (time,) or (batch, time) tensor
        frame_size: n_fft / frame length
        hop_length: samples between adjacent frames
        window: precomputed window tensor of shape (frame_size,). Defaults
            to a periodic Hann window (the NumPy path's default).
        center: pad ``frame_size // 2`` on both sides before framing
        pad_mode: 'constant' or 'reflect' (reflect falls back to constant
            when the signal is too short, matching the NumPy path)

    Returns:
        Complex tensor of shape (n_freq, n_frames) or (batch, n_freq,
        n_frames), n_freq = frame_size // 2 + 1.
    """
    x, unbatched = _as_batch(signal)
    if x.shape[-1] == 0:
        raise ValueError('stft received an empty signal')

    if window is None:
        window = hann_window(frame_size, device=x.device, dtype=x.dtype)

    if center:
        pad = frame_size // 2
        if pad_mode == 'reflect' and x.shape[-1] > pad:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (pad, pad), mode='reflect'
            ).squeeze(1)
        else:
            x = torch.nn.functional.pad(x, (pad, pad), mode='constant')

    if x.shape[-1] < frame_size:
        x = torch.nn.functional.pad(
            x, (0, frame_size - x.shape[-1]), mode='constant'
        )

    # (batch, n_frames, frame_size), the same framing as
    # sliding_window_view(signal, frame_size)[::hop_length]
    frames = x.unfold(-1, frame_size, hop_length)
    spec = torch.fft.rfft(frames * window, n=frame_size, dim=-1)
    spec = spec.transpose(-1, -2)  # (batch, n_freq, n_frames)
    return _unbatch(spec, unbatched)


def magnitude_spectrogram(stft_matrix: torch.Tensor) -> torch.Tensor:
    """Magnitude spectrogram |X|."""
    return torch.abs(stft_matrix)


def power_spectrogram(stft_matrix: torch.Tensor) -> torch.Tensor:
    """Power spectrogram |X|^2."""
    return torch.abs(stft_matrix) ** 2


def mel_filterbank(
    n_mels: int,
    n_fft: int,
    sr: int = 22050,
    fmin: float = 0.0,
    fmax: float | None = None,
    norm: str = 'slaney',
    htk: bool = False,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Mel filterbank as a torch tensor of shape (n_mels, n_fft // 2 + 1).

    Built with the exact NumPy construction (``afx_audio.methods.mel``) and
    converted, so it is bit-identical to the CPU path up to dtype casting.
    The filterbank is a constant: it carries no gradient, but gradients
    flow *through* it to the spectrogram.
    """
    fb = _np_mel_filterbank(
        n_mels, n_fft, sr=sr, fmin=fmin, fmax=fmax, norm=norm, htk=htk
    )
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.as_tensor(fb, device=device, dtype=dtype)


def dct_matrix(
    n_out: int, n_in: int, *, device=None, dtype=None
) -> torch.Tensor:
    """
    DCT-II matrix with orthogonal normalization, shape (n_out, n_in), such
    that ``dct_matrix(k, N) @ x == scipy.fft.dct(x, type=2,
    norm='ortho', axis=0)[:k]``.
    """
    n = np.arange(n_in)
    k = np.arange(n_out)[:, np.newaxis]
    mat = 2.0 * np.cos(np.pi * k * (2.0 * n + 1.0) / (2.0 * n_in))
    # Orthogonal normalization (matches scipy norm='ortho')
    scale = np.full((n_out, 1), np.sqrt(1.0 / (2.0 * n_in)))
    if n_out > 0:
        scale[0, 0] = np.sqrt(1.0 / (4.0 * n_in))
    mat = mat * scale
    if dtype is None:
        dtype = torch.get_default_dtype()
    return torch.as_tensor(mat, device=device, dtype=dtype)


def log_mel_spectrogram(
    mel_spec: torch.Tensor,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> torch.Tensor:
    """
    dB-scaled mel spectrogram, mirroring
    ``afx_audio.methods.mel.log_mel_spectrogram``. For batched input the
    ``top_db`` dynamic-range floor is applied per sample.
    """
    log_mel = 10.0 * torch.log10(torch.clamp(mel_spec, min=amin))
    if top_db is not None:
        if log_mel.dim() >= 3:
            ref = log_mel.amax(dim=(-2, -1), keepdim=True)
        else:
            ref = log_mel.amax()
        log_mel = torch.maximum(log_mel, ref - top_db)
    return log_mel


# ---------------------------------------------------------------------------
# Feature extractors (parity with afx_audio.extractors defaults)
# ---------------------------------------------------------------------------

def melspectrogram(
    signal: torch.Tensor,
    sr: int,
    *,
    n_mels: int = 128,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    window: torch.Tensor | None = None,
    mel_fb: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Mel-scaled power spectrogram, (..., n_mels, n_frames). Mirrors
    ``afx_audio.extractors.cepstral_features.extract_melspectrogram``.
    """
    spec = stft(
        signal, frame_size=frame_size, hop_length=hop_length,
        window=window, center=center,
    )
    power = power_spectrogram(spec)
    if mel_fb is None:
        mel_fb = mel_filterbank(
            n_mels, frame_size, sr=sr, fmin=fmin, fmax=fmax,
            device=power.device, dtype=power.dtype,
        )
    return mel_fb @ power


def mfcc(
    signal: torch.Tensor,
    sr: int,
    *,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    center: bool = True,
    top_db: float | None = 80.0,
    window: torch.Tensor | None = None,
    mel_fb: torch.Tensor | None = None,
    dct_mat: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MFCCs, (..., n_mfcc, n_frames). Mirrors
    ``afx_audio.extractors.cepstral_features.extract_mfcc``: STFT -> power -> mel (slaney) -> dB (top_db below each
    clip's peak) -> DCT-II. Fully differentiable.
    """
    mel_spec = melspectrogram(
        signal, sr, n_mels=n_mels, frame_size=frame_size,
        hop_length=hop_length, fmin=fmin, fmax=fmax, center=center,
        window=window, mel_fb=mel_fb,
    )
    log_mel = log_mel_spectrogram(mel_spec, amin=1e-10, top_db=top_db)
    if dct_mat is None:
        dct_mat = dct_matrix(
            n_mfcc, log_mel.shape[-2],
            device=log_mel.device, dtype=log_mel.dtype,
        )
    return dct_mat @ log_mel


def _mag_and_freqs(
    signal: torch.Tensor,
    sr: int,
    frame_size: int,
    hop_length: int,
    center: bool,
    window: torch.Tensor | None,
    pad_mode: str = 'constant',
) -> tuple[torch.Tensor, torch.Tensor]:
    spec = stft(
        signal, frame_size=frame_size, hop_length=hop_length,
        window=window, center=center, pad_mode=pad_mode,
    )
    mag = torch.abs(spec)
    freqs = _rfft_freqs(frame_size, sr, device=mag.device, dtype=mag.dtype)
    return mag, freqs


def spectral_centroid(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Magnitude-weighted mean frequency, (..., n_frames)."""
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                                window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > 0, msum, torch.ones_like(msum))
    weighted = (mag * freqs.unsqueeze(-1)).sum(dim=-2)
    return torch.where(msum > 0, weighted / safe, torch.zeros_like(msum))


def spectral_bandwidth(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Magnitude-weighted std of frequency around the centroid."""
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                                window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > 0, msum, torch.ones_like(msum))
    centroid = (mag * freqs.unsqueeze(-1)).sum(dim=-2) / safe
    dev2 = (freqs.unsqueeze(-1) - centroid.unsqueeze(-2)) ** 2
    var = (dev2 * mag).sum(dim=-2) / safe
    return torch.where(
        msum > 0, _safe_sqrt(var), torch.zeros_like(msum)
    )


def spectral_rolloff(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
    center: bool = True,
    window: torch.Tensor | None = None,
    power: float = 1.0,
) -> torch.Tensor:
    """
    Lowest bin frequency where the cumulative sum of |X|^power reaches
    ``roll_percent`` of the total (power=1 as in librosa). Index
    selection, so not differentiable w.r.t. the waveform.
    """
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                                window)
    cum = torch.cumsum(mag if power == 1 else mag ** power, dim=-2)
    target = roll_percent * cum.select(-2, cum.shape[-2] - 1)
    # First bin where cum >= target == count of bins where cum < target
    # (cum is nondecreasing along the frequency axis).
    idx = torch.clamp((cum < target.unsqueeze(-2)).sum(dim=-2), max=cum.shape[-2] - 1)
    return freqs[idx]


def spectral_entropy(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Shannon entropy (bits) of the normalized magnitude spectrum; NaN when silent."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > _EPS, msum, torch.ones_like(msum))
    ps = mag / safe.unsqueeze(-2)
    entropy = -(ps * torch.log2(ps + _EPS)).sum(dim=-2)
    return torch.where(msum > _EPS, entropy, torch.full_like(entropy, float('nan')))


def spectral_flatness(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Geometric / arithmetic mean ratio of the power spectrum."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window)
    power = torch.clamp(mag ** 2, min=_EPS)
    return torch.exp(torch.log(power).mean(dim=-2)) / power.mean(dim=-2)


def spectral_flux(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Euclidean distance between consecutive magnitude spectra,
    (..., n_frames - 1). Mirrors extract_spectral_flux (reflect
    padding when centered).
    """
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window, pad_mode='reflect')
    if mag.shape[-1] < 2:
        return torch.zeros(mag.shape[:-2] + (1,), device=mag.device, dtype=mag.dtype)
    diffs = mag.diff(dim=-1)
    return _safe_sqrt((diffs ** 2).sum(dim=-2))


def _time_frames(x: torch.Tensor, frame_size: int | None, hop_length: int | None,
                 center: bool, pad_mode: str = 'constant') -> torch.Tensor:
    """(batch, n_frames, frame_len) frames with the rules of afx_audio.utils.frame_utils.analysis_frames."""
    if frame_size is None or hop_length is None:
        return x.unsqueeze(-2)
    if center:
        pad = frame_size // 2
        if pad_mode == 'edge':
            x = torch.nn.functional.pad(x.unsqueeze(1), (pad, pad), mode='replicate').squeeze(1)
        else:
            x = torch.nn.functional.pad(x, (pad, pad), mode='constant')
    if x.shape[-1] < frame_size:
        return x.unsqueeze(-2)
    return x.unfold(-1, frame_size, hop_length)


def zero_crossing_rate(
    signal: torch.Tensor,
    sr: int | None = None,
    *,
    frame_size: int | None = 2048,
    hop_length: int | None = 512,
    center: bool = True,
    threshold: float = 1e-10,
) -> torch.Tensor:
    """
    Zero-crossing rate per frame, (..., n_frames): sign changes over the
    frame length, |x| <= threshold counted as zero, centered frames padded
    by repeating the edges. Counting is not differentiable (gradient is
    zero almost everywhere).
    """
    x, unbatched = _as_batch(signal)
    frames = _time_frames(x, frame_size, hop_length, center, pad_mode='edge')
    negative = (frames < -threshold).to(x.dtype)
    zcr = negative.diff(dim=-1).abs().sum(dim=-1) / frames.shape[-1]
    return _unbatch(zcr, unbatched)


def rms_energy(
    signal: torch.Tensor,
    sr: int | None = None,
    *,
    frame_size: int | None = None,
    hop_length: int | None = None,
    center: bool = True,
) -> torch.Tensor:
    """RMS per frame, (..., n_frames); the whole clip as one frame without frame sizes. Differentiable."""
    x, unbatched = _as_batch(signal)
    frames = _time_frames(x, frame_size, hop_length, center)
    return _unbatch(_safe_sqrt((frames ** 2).mean(dim=-1)), unbatched)


def spectral_skewness(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Third standardized moment of frequency under the magnitude spectrum, (..., n_frames); NaN when silent."""
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > 0, msum, torch.ones_like(msum))
    centroid = (mag * freqs.unsqueeze(-1)).sum(dim=-2) / safe
    dev = freqs.unsqueeze(-1) - centroid.unsqueeze(-2)
    bandwidth = _safe_sqrt((dev ** 2 * mag).sum(dim=-2) / safe)
    safe_bw = torch.where(bandwidth > 0, bandwidth, torch.ones_like(bandwidth))
    z = dev / safe_bw.unsqueeze(-2)
    skew = (z ** 3 * mag).sum(dim=-2) / safe
    return torch.where((msum > 0) & (bandwidth > 0), skew, torch.full_like(skew, float('nan')))


def spectral_slope(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares slope of 20 log10(|X| + 1e-10) against frequency in dB/Hz, (..., n_frames); NaN when silent."""
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    mag_db = 20.0 * torch.log10(mag + _EPS)
    x = freqs - freqs.mean()
    slope = (x.unsqueeze(-1) * mag_db).sum(dim=-2) / (x ** 2).sum()
    return torch.where((mag > 0).any(dim=-2), slope, torch.full_like(slope, float('nan')))


def spectral_kurtosis(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    fisher: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fourth standardized moment of frequency under the magnitude spectrum, (..., n_frames); NaN when silent."""
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > 0, msum, torch.ones_like(msum))
    centroid = (mag * freqs.unsqueeze(-1)).sum(dim=-2) / safe
    dev2 = (freqs.unsqueeze(-1) - centroid.unsqueeze(-2)) ** 2
    var = (dev2 * mag).sum(dim=-2) / safe
    defined = (msum > 0) & (var > 0)
    safe_var = torch.where(defined, var, torch.ones_like(var))
    kurt = (dev2 * dev2 * mag).sum(dim=-2) / safe / safe_var ** 2
    kurt = torch.where(defined, kurt, torch.full_like(kurt, float('nan')))
    return kurt - 3.0 if fisher else kurt


def spectral_crest(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Largest over mean magnitude of the spectrum, (..., n_frames); NaN when silent."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    mean = mag.mean(dim=-2)
    positive = mean > 0
    ratio = mag.amax(dim=-2) / torch.where(positive, mean, torch.ones_like(mean))
    return torch.where(positive, ratio, torch.full_like(ratio, float('nan')))


def spectral_decrease(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """Spectral decrease (Peeters, 2004) with the DC bin as reference, (..., n_frames); NaN when silent."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    k = torch.arange(1, mag.shape[-2], device=mag.device, dtype=mag.dtype).unsqueeze(-1)
    num = ((mag[..., 1:, :] - mag[..., :1, :]) / k).sum(dim=-2)
    den = mag[..., 1:, :].sum(dim=-2)
    positive = den > 0
    ratio = num / torch.where(positive, den, torch.ones_like(den))
    return torch.where(positive, ratio, torch.full_like(ratio, float('nan')))


def _contrast_bands(sr: int, frame_size: int, n_bands: int, fmin: float, quantile: float):
    """Bin indices and quantile size per band, as in extract_spectral_contrast."""
    freq = np.fft.rfftfreq(frame_size, d=1.0 / sr)
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))
    if np.any(octa[:-1] >= sr / 2.0):
        raise ValueError('Frequency band exceeds Nyquist. Reduce fmin or n_bands.')
    bands: list[tuple[np.ndarray, int] | None] = []
    for k in range(n_bands + 1):
        band = np.logical_and(freq >= octa[k], freq <= octa[k + 1])
        idx = np.flatnonzero(band)
        if len(idx) == 0:
            bands.append(None)
            continue
        if k > 0 and idx[0] > 0:
            band[idx[0] - 1] = True
        if k == n_bands:
            band[np.flatnonzero(band)[-1] + 1:] = True
        idx = np.flatnonzero(band)
        n_quantile = int(max(np.rint(quantile * len(idx)), 1))
        if k < n_bands:
            idx = idx[:-1]
        bands.append((idx, n_quantile) if len(idx) else None)
    return bands


def _power_to_db(x: torch.Tensor, amin: float = 1e-10, top_db: float = 80.0) -> torch.Tensor:
    db = 10.0 * torch.log10(torch.clamp(x, min=amin))
    ref = db.amax(dim=(-2, -1), keepdim=True)
    return torch.maximum(db, ref - top_db)


def spectral_contrast(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    center: bool = True,
    window: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Spectral contrast, (..., n_bands + 1, n_frames), as
    extract_spectral_contrast (librosa's definition). Gradients flow
    through the sorted magnitudes.
    """
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center, window)
    peaks, valleys = [], []
    zeros = torch.zeros(mag.shape[:-2] + (mag.shape[-1],), device=mag.device, dtype=mag.dtype)
    for band in _contrast_bands(sr, frame_size, n_bands, fmin, quantile):
        if band is None:
            peaks.append(zeros)
            valleys.append(zeros)
            continue
        idx, nq = band
        sub = torch.sort(mag[..., torch.as_tensor(idx, device=mag.device), :], dim=-2).values
        valleys.append(sub[..., :nq, :].mean(dim=-2))
        peaks.append(sub[..., -nq:, :].mean(dim=-2))
    return _power_to_db(torch.stack(peaks, dim=-2)) - _power_to_db(torch.stack(valleys, dim=-2))


def chroma_filterbank(sr: int, n_fft: int, tuning: float = 0.0, *, device=None, dtype=None) -> torch.Tensor:
    """librosa's chroma filterbank (afx_audio.methods.chroma.chroma_filterbank) as a tensor, (12, n_fft // 2 + 1)."""
    fb = _np_chroma_filterbank(sr, n_fft, tuning=tuning)
    return torch.as_tensor(fb, device=device, dtype=dtype or torch.get_default_dtype())


def chroma_stft(
    signal: torch.Tensor,
    sr: int,
    *,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    tuning: float = 0.0,
    normalize: bool = True,
    window: torch.Tensor | None = None,
    chroma_fb: torch.Tensor | None = None,
) -> torch.Tensor:
    """Chromagram, (..., 12, n_frames), as extract_chroma_stft (librosa.feature.chroma_stft at fixed tuning)."""
    power = power_spectrogram(stft(signal, frame_size=frame_size, hop_length=hop_length,
                                   window=window, center=center))
    if chroma_fb is None:
        chroma_fb = chroma_filterbank(sr, frame_size, tuning, device=power.device, dtype=power.dtype)
    chroma = chroma_fb @ power
    if not normalize:
        return chroma
    peak = chroma.abs().amax(dim=-2, keepdim=True)
    tiny = torch.finfo(chroma.dtype).tiny
    return chroma / torch.where(peak < tiny, torch.ones_like(peak), peak)


def _savgol_parts(width: int, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Interior kernel and the (width, width) operator whose first/last rows give the 'interp' edges."""
    from scipy.signal import savgol_filter

    edge = savgol_filter(np.eye(width), width, polyorder=order, deriv=order, mode='interp', axis=0)
    return edge[width // 2], edge


def delta(features: torch.Tensor, width: int = 9, order: int = 1) -> torch.Tensor:
    """
    Delta features along the last axis, equal to afx_audio.methods.delta.compute_delta
    (librosa.feature.delta): Savitzky-Golay derivative with polynomial edge
    fits. ``width`` is clipped to the largest odd value not above the number
    of frames; fewer than 3 frames give zeros. Differentiable.
    """
    if order not in (1, 2):
        raise ValueError(f'order must be 1 or 2, got {order}')
    n = features.shape[-1]
    width = min(width, n)
    if width % 2 == 0:
        width -= 1
    if width < 3:
        return torch.zeros_like(features)
    kernel_np, edge_np = _savgol_parts(width, order)
    kernel = torch.as_tensor(kernel_np, device=features.device, dtype=features.dtype)
    edge = torch.as_tensor(edge_np, device=features.device, dtype=features.dtype)
    half = width // 2
    lead = features.shape[:-1]
    flat = features.reshape(-1, 1, n)
    interior = torch.nn.functional.conv1d(flat, kernel.view(1, 1, -1)).reshape(*lead, n - 2 * half)
    left = features[..., :width] @ edge[:half].T
    right = features[..., -width:] @ edge[width - half:].T
    return torch.cat([left, interior, right], dim=-1)


def gfcc(
    signal: torch.Tensor,
    sr: int,
    *,
    n_gfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_gammatone: int = 64,
    fmin: float = 50.0,
    fmax: float | None = None,
    center: bool = True,
    window: torch.Tensor | None = None,
    gammatone_fb: torch.Tensor | None = None,
    dct_mat: torch.Tensor | None = None,
) -> torch.Tensor:
    """GFCC, (..., n_gfcc, n_frames), as extract_gfcc. Differentiable."""
    power = power_spectrogram(stft(signal, frame_size=frame_size, hop_length=hop_length,
                                   window=window, center=center))
    if gammatone_fb is None:
        fb = _np_gammatone_filterbank(n_gammatone, frame_size, sr, fmin, sr / 2.0 if fmax is None else fmax)
        gammatone_fb = torch.as_tensor(np.array(fb), device=power.device, dtype=power.dtype)
    log_spec = log_mel_spectrogram(gammatone_fb @ power, amin=1e-10, top_db=80.0)
    if dct_mat is None:
        dct_mat = dct_matrix(n_gfcc, log_spec.shape[-2], device=log_spec.device, dtype=log_spec.dtype)
    return dct_mat @ log_spec
