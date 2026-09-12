"""
Differentiable, GPU-capable feature transforms implemented in pure PyTorch.

These functions mirror the NumPy implementations in ``AFX.methods`` and
``AFX.extractors`` (same windowing, padding, filterbank and normalization
conventions), so numerical results match the CPU/NumPy path to floating
point tolerance. All functions:

- accept a 1-D ``(time,)`` or batched 2-D ``(batch, time)`` waveform tensor,
- run on whatever device/dtype the input tensor lives on (CPU or CUDA),
- are composed of differentiable ops, so gradients flow back to the
  waveform (except where noted, e.g. ``zero_crossing_rate`` and
  ``spectral_rolloff`` which involve inherently non-differentiable
  counting/selection).

Constant tensors (windows, mel filterbanks, DCT matrices) can be
precomputed once and passed in — ``AFX.torch.AFXFeatureModule`` does this
and registers them as buffers so they move with ``.to(device)``.
"""
from typing import Optional, Tuple

import numpy as np
import torch

from AFX.methods.mel import mel_filterbank as _np_mel_filterbank

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
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_batch(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """Promote (time,) to (1, time). Returns (tensor, was_unbatched)."""
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
    window: Optional[torch.Tensor] = None,
    center: bool = True,
    pad_mode: str = 'constant',
) -> torch.Tensor:
    """
    Short-Time Fourier Transform, mirroring ``AFX.methods.stft.stft``.

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

    # (batch, n_frames, frame_size) — same framing as
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
    fmax: Optional[float] = None,
    norm: str = 'slaney',
    htk: bool = False,
    *,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Mel filterbank as a torch tensor of shape (n_mels, n_fft // 2 + 1).

    Built with the exact NumPy construction (``AFX.methods.mel``) and
    converted, so it is bit-identical to the CPU path up to dtype casting.
    The filterbank is a constant — it carries no gradient, but gradients
    flow *through* it to the spectrogram.
    """
    fb = _np_mel_filterbank(
        n_mels, n_fft, sr=sr, fmin=fmin, fmax=fmax, norm=norm, htk=htk
    )
    if dtype is None:
        dtype = torch.float32
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
        dtype = torch.float32
    return torch.as_tensor(mat, device=device, dtype=dtype)


def log_mel_spectrogram(
    mel_spec: torch.Tensor,
    amin: float = 1e-10,
    top_db: Optional[float] = 80.0,
) -> torch.Tensor:
    """
    dB-scaled mel spectrogram, mirroring
    ``AFX.methods.mel.log_mel_spectrogram``. For batched input the
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
# Feature extractors (parity with AFX.extractors defaults)
# ---------------------------------------------------------------------------

def melspectrogram(
    signal: torch.Tensor,
    sr: int,
    n_mels: int = 128,
    frame_size: int = 2048,
    hop_length: int = 512,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    center: bool = True,
    window: Optional[torch.Tensor] = None,
    mel_fb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Mel-scaled power spectrogram, (..., n_mels, n_frames). Mirrors
    ``AFX.extractors.cepstral_features.extract_melspectrogram``
    (center=True).
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
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
    mel_fb: Optional[torch.Tensor] = None,
    dct_mat: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    MFCCs, (..., n_mfcc, n_frames). Mirrors
    ``AFX.extractors.cepstral_features.extract_mfcc`` (center=False
    default): STFT -> power -> mel (slaney) -> dB (top_db=80) -> DCT-II.
    Fully differentiable.
    """
    mel_spec = melspectrogram(
        signal, sr, n_mels=n_mels, frame_size=frame_size,
        hop_length=hop_length, fmin=fmin, fmax=fmax, center=center,
        window=window, mel_fb=mel_fb,
    )
    log_mel = log_mel_spectrogram(mel_spec, amin=1e-10, top_db=80.0)
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
    window: Optional[torch.Tensor],
    pad_mode: str = 'constant',
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
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
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
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
        msum > 0, torch.sqrt(var), torch.zeros_like(msum)
    )


def spectral_rolloff(
    signal: torch.Tensor,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Frequency below which ``roll_percent`` of spectral energy lies
    (linearly interpolated). Involves index selection, so it is not
    differentiable w.r.t. the waveform.
    """
    mag, freqs = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                                window)
    power = mag ** 2
    cum = torch.cumsum(power, dim=-2)
    total = cum.select(-2, cum.shape[-2] - 1)
    target = roll_percent * total
    # First bin where cum >= target == count of bins where cum < target
    # (cum is nondecreasing along the frequency axis).
    idx = (cum < target.unsqueeze(-2)).sum(dim=-2)
    idx = torch.clamp(idx, max=cum.shape[-2] - 1)
    prev_idx = torch.clamp(idx - 1, min=0)

    e_before = torch.gather(cum, -2, prev_idx.unsqueeze(-2)).squeeze(-2)
    e_after = torch.gather(cum, -2, idx.unsqueeze(-2)).squeeze(-2)
    f_before = freqs[prev_idx]
    f_after = freqs[idx]
    denom = torch.where(
        e_after > e_before, e_after - e_before, torch.ones_like(e_after)
    )
    alpha = torch.clamp((target - e_before) / denom, 0.0, 1.0)
    interp = f_before + alpha * (f_after - f_before)
    rolloff = torch.where(idx > 0, interp, freqs[idx])
    return torch.where(total > 0, rolloff, torch.zeros_like(rolloff))


def spectral_entropy(
    signal: torch.Tensor,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shannon entropy (bits) of the normalized magnitude spectrum."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window)
    msum = mag.sum(dim=-2)
    safe = torch.where(msum > _EPS, msum, torch.ones_like(msum))
    ps = mag / safe.unsqueeze(-2)
    entropy = -(ps * torch.log2(ps + _EPS)).sum(dim=-2)
    return torch.where(msum > _EPS, entropy, torch.zeros_like(entropy))


def spectral_flatness(
    signal: torch.Tensor,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Geometric / arithmetic mean ratio of the power spectrum."""
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window)
    power = torch.clamp(mag ** 2, min=_EPS)
    geometric = torch.exp(torch.log(power).mean(dim=-2))
    arithmetic = power.mean(dim=-2)
    return torch.where(
        arithmetic > _EPS, geometric / arithmetic,
        torch.zeros_like(arithmetic)
    )


def spectral_flux(
    signal: torch.Tensor,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = True,
    window: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Euclidean distance between consecutive magnitude spectra,
    (..., n_frames - 1). Mirrors extract_spectral_flux (center=True,
    reflect padding).
    """
    mag, _ = _mag_and_freqs(signal, sr, frame_size, hop_length, center,
                            window, pad_mode='reflect')
    diffs = mag.diff(dim=-1)
    return torch.sqrt((diffs ** 2).sum(dim=-2))


def zero_crossing_rate(
    signal: torch.Tensor,
    sr: Optional[int] = None,
    frame_size: int = 2048,
    hop_length: int = 512,
) -> torch.Tensor:
    """
    Zero-crossing rate per frame, (..., n_frames). Counting sign changes
    is inherently non-differentiable (gradient is zero a.e.).
    """
    x, unbatched = _as_batch(signal)
    n = x.shape[-1]
    if n < frame_size:
        if n <= 1:
            zcr = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        else:
            signs = (x < 0).to(x.dtype)
            crossings = signs.diff(dim=-1).abs().sum(dim=-1, keepdim=True)
            zcr = crossings / (n - 1)
        return _unbatch(zcr, unbatched)

    frames = x.unfold(-1, frame_size, hop_length)  # (B, n_frames, frame)
    signs = (frames < 0).to(x.dtype)
    crossings = signs.diff(dim=-1).abs().sum(dim=-1)
    zcr = crossings / (frame_size - 1)
    return _unbatch(zcr, unbatched)


def rms_energy(
    signal: torch.Tensor,
    sr: Optional[int] = None,
) -> torch.Tensor:
    """RMS energy of the whole signal, shape (..., 1). Differentiable."""
    x, unbatched = _as_batch(signal)
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))
    return _unbatch(rms, unbatched)
