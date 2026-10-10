"""
eGeMAPS v02, the extended Geneva Minimalistic Acoustic Parameter Set.

Eyben, Scherer, Schuller, Sundberg, Andre, Busso, Devillers, Epps, Laukka,
Narayanan and Truong (2016), "The Geneva Minimalistic Acoustic Parameter Set
(GeMAPS) for Voice Research and Affective Computing", IEEE Transactions on
Affective Computing 7(2), 190-202.

Written from the paper and the parameters documented in openSMILE's
eGeMAPSv02 configuration (frame lengths, windows, band edges, filter counts,
smoothing, functionals). Where openSMILE's output departs from the paper's
prose, the openSMILE behaviour is followed so values are comparable with
published eGeMAPS numbers; docs/egemaps.md lists those cases and the
measured agreement.

All descriptors use a 10 ms hop. Frame i covers samples [i * hop, i * hop +
length) of the signal: 20 ms Hamming frames for loudness, spectral shape,
MFCC and formants, 60 ms Gaussian frames (sigma 0.4) for F0, jitter,
shimmer, HNR and harmonic levels.
"""
from typing import Any

import numpy as np
from scipy.interpolate import CubicSpline

from afx_audio.methods.lpc import levinson, lpc_roots
from afx_audio.utils.validation import validate_signal, validate_sr

__all__ = [
    'EGEMAPS_LLD_NAMES',
    'EGEMAPS_NAMES',
    'extract_egemaps_lld',
    'extract_egemaps',
    'egemaps',
]

EGEMAPS_LLD_NAMES = (
    'Loudness_sma3', 'alphaRatio_sma3', 'hammarbergIndex_sma3', 'slope0-500_sma3', 'slope500-1500_sma3',
    'spectralFlux_sma3', 'mfcc1_sma3', 'mfcc2_sma3', 'mfcc3_sma3', 'mfcc4_sma3',
    'F0semitoneFrom27.5Hz_sma3nz', 'jitterLocal_sma3nz', 'shimmerLocaldB_sma3nz', 'HNRdBACF_sma3nz',
    'logRelF0-H1-H2_sma3nz', 'logRelF0-H1-A3_sma3nz',
    'F1frequency_sma3nz', 'F1bandwidth_sma3nz', 'F1amplitudeLogRelF0_sma3nz',
    'F2frequency_sma3nz', 'F2bandwidth_sma3nz', 'F2amplitudeLogRelF0_sma3nz',
    'F3frequency_sma3nz', 'F3bandwidth_sma3nz', 'F3amplitudeLogRelF0_sma3nz',
)

_CONTOUR = ('amean', 'stddevNorm', 'percentile20.0', 'percentile50.0', 'percentile80.0', 'pctlrange0-2',
            'meanRisingSlope', 'stddevRisingSlope', 'meanFallingSlope', 'stddevFallingSlope')
_VOICED = ('jitterLocal', 'shimmerLocaldB', 'HNRdBACF', 'logRelF0-H1-H2', 'logRelF0-H1-A3',
           'F1frequency', 'F1bandwidth', 'F1amplitudeLogRelF0', 'F2frequency', 'F2bandwidth',
           'F2amplitudeLogRelF0', 'F3frequency', 'F3bandwidth', 'F3amplitudeLogRelF0',
           'alphaRatioV', 'hammarbergIndexV', 'slopeV0-500', 'slopeV500-1500', 'spectralFluxV',
           'mfcc1V', 'mfcc2V', 'mfcc3V', 'mfcc4V')
_UNVOICED = ('alphaRatioUV', 'hammarbergIndexUV', 'slopeUV0-500', 'slopeUV500-1500', 'spectralFluxUV')

EGEMAPS_NAMES = (
    *(f'F0semitoneFrom27.5Hz_sma3nz_{f}' for f in _CONTOUR),
    *(f'loudness_sma3_{f}' for f in _CONTOUR),
    *(f'{d}_sma3_{f}' for d in ('spectralFlux', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4') for f in ('amean', 'stddevNorm')),
    *(f'{d}_sma3nz_{f}' for d in _VOICED for f in ('amean', 'stddevNorm')),
    *(f'{d}_sma3nz_amean' for d in _UNVOICED),
    'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'StddevVoicedSegmentLengthSec',
    'MeanUnvoicedSegmentLength', 'StddevUnvoicedSegmentLength', 'equivalentSoundLevel_dBp',
)

_HOP = 0.010
_SHORT = 0.020
_LONG = 0.060
_TINY = 1e-30
_BLOCK = 1000

# Pitch search range and SHS settings of the eGeMAPSv02 configuration (Hermes, 1988)
_F0_MIN, _F0_MAX = 55.0, 1000.0
_SHS_HARMONICS = 15
_SHS_COMPRESSION = 0.85
_SHS_POINTS_PER_OCTAVE = 48
_SHS_CANDIDATES = 6
# Frames whose Gaussian-windowed RMS is below this are unvoiced (openSMILE threshold, full scale = 1)
_VOICING_RMS = 0.001
# Voicing thresholds and run lengths, tuned for frame agreement with openSMILE's voicing on
# LibriSpeech and synthetic speech (91.5 % and 98.8 % of frames)
_VOICING_NCCF = 0.3
_VOICING_SHS = 8.0
_MAX_GAP_FRAMES = 2
_MIN_VOICED_FRAMES = 3
_FORMANT_MAX_BANDWIDTH = 1000.0
# Scale of the compressed auditory spectrum sum that matches openSMILE's loudness units
_LOUDNESS_GAIN = 0.0690


def _next_pow2(n: int) -> int:
    return 1 << max(0, int(n - 1).bit_length())


def _frames(x: np.ndarray, length: int, hop: int, n_frames: int) -> np.ndarray:
    pad = max(0, (n_frames - 1) * hop + length - len(x))
    x = np.concatenate([x, np.zeros(pad)]) if pad else x
    return np.lib.stride_tricks.sliding_window_view(x, length)[::hop][:n_frames]


def _hamming(n: int) -> np.ndarray:
    return 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / max(n - 1, 1))


def _gauss(n: int, sigma: float = 0.4) -> np.ndarray:
    half = max(n - 1, 1) / 2
    return np.exp(-0.5 * ((np.arange(n) - half) / (sigma * half)) ** 2)


def _db(power: np.ndarray) -> np.ndarray:
    return 10 * np.log10(np.maximum(power, _TINY))


def _htk_mel(f):
    return 1127.0 * np.log1p(np.asarray(f, dtype=np.float64) / 700.0)


def _mel_bank(freqs: np.ndarray, n_bands: int, fmin: float, fmax: float) -> tuple[np.ndarray, np.ndarray]:
    """Unit-peak triangles equally spaced on the HTK mel scale, and their centre frequencies in Hz."""
    edges = np.linspace(_htk_mel(fmin), _htk_mel(fmax), n_bands + 2)
    m = _htk_mel(freqs)
    left, centre, right = edges[:-2, None], edges[1:-1, None], edges[2:, None]
    bank = np.maximum(0.0, np.minimum((m - left) / (centre - left), (right - m) / (right - centre)))
    return bank, 700.0 * np.expm1(edges[1:-1] / 1127.0)


def _equal_loudness(f: np.ndarray) -> np.ndarray:
    """
    Equal-loudness weight of Hermansky (1990), "Perceptual linear predictive (PLP) analysis of speech", eq. 4.

    E(w) = (w^2 + 56.8e6) w^4 / ((w^2 + 6.3e6)^2 (w^2 + 0.38e9)), w = 2 pi f,
    times a roll-off 1 / (1 + (f / 4580 Hz)^6.88) in place of Hermansky's
    high-frequency term. The roll-off constants were fitted to openSMILE's
    loudness of pure tones (40 Hz to 7.9 kHz); with Hermansky's term the
    loudness of tones above 4 kHz is off by more than 20 %.
    """
    w2 = (2 * np.pi * f) ** 2
    return (w2 + 56.8e6) * w2 ** 2 / ((w2 + 6.3e6) ** 2 * (w2 + 0.38e9)) / (1 + (f / 4580.0) ** 6.88)


def _band_slope(db: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """
    Regression slope of the dB spectrum over lo <= f <= hi, in dB/Hz.

    openSMILE evaluates (N' sum(f y) - sum f sum y) / (N' sum f^2 - (sum f)^2)
    with N' = N - 1 for N bins, which is not the least-squares slope; this
    reproduces it.
    """
    sel = (freqs >= lo) & (freqs <= hi)
    n = sel.sum() - 1
    if n < 1:
        return np.zeros(len(db))
    f = freqs[sel]
    y = db[:, sel]
    den = n * (f ** 2).sum() - f.sum() ** 2
    return (n * (y @ f) - f.sum() * y.sum(axis=1)) / den if den != 0 else np.zeros(len(db))


def _spectral_llds(x: np.ndarray, sr: float, n_frames: int) -> dict[str, np.ndarray]:
    """Loudness, spectral balance, flux, MFCC 1-4, formants and frame power on 20 ms Hamming frames."""
    hop = int(round(_HOP * sr))
    length = int(round(_SHORT * sr))
    raw = _frames(x, length, hop, n_frames)
    nfft = _next_pow2(length)
    spec = np.fft.rfft(raw * _hamming(length), nfft)
    mag = np.abs(spec)
    power = mag ** 2
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    out: dict[str, np.ndarray] = {'power': np.mean(raw ** 2, axis=1)}

    bank, centres = _mel_bank(freqs, 26, 20.0, min(8000.0, sr / 2))
    bands = power @ bank.T
    out['loudness'] = _LOUDNESS_GAIN * ((bands * _equal_loudness(centres)) ** 0.33).sum(axis=1)

    # Band edges as openSMILE applies them: the 50 Hz lower edge of the paper is not honoured
    low, high = power[:, freqs < 1000].sum(1), power[:, (freqs >= 1000) & (freqs <= 5000)].sum(1)
    ok = (low > 0) & (high > 0)
    out['alpha'] = np.where(ok, 10 * np.log10(np.where(ok, high, 1) / np.where(ok, low, 1)), 0.0)
    low = power[:, freqs < 2000].max(1)
    high = power[:, (freqs >= 2000) & (freqs <= 5000)].max(1) if (freqs >= 2000).any() else np.zeros(n_frames)
    ok = (low > 0) & (high > 0)
    out['hammarberg'] = np.where(ok, 10 * np.log10(np.where(ok, low, 1) / np.where(ok, high, 1)), 0.0)

    db = _db(power)
    silent = power.max(axis=1) == 0
    out['slope_low'] = np.where(silent, 0.0, _band_slope(db, freqs, 0, 500))
    out['slope_high'] = np.where(silent, 0.0, _band_slope(db, freqs, 500, 1500))

    flux_mag = mag[:, freqs < 5000]
    out['flux'] = np.sqrt(np.mean(np.diff(flux_mag, axis=0, prepend=flux_mag[:1]) ** 2, axis=1))

    # HTK-style MFCC: log mel energies floored at one 16-bit quantization step, DCT-II, sinusoidal lifter L = 22
    logmel = np.log(np.maximum(bands, 1.0 / 32768 ** 2))
    k = np.arange(1, 5)[:, None]
    dct = np.sqrt(2.0 / 26) * np.cos(np.pi * k * (np.arange(26)[None, :] + 0.5) / 26)
    out['mfcc'] = (logmel @ dct.T) * (1 + 11 * np.sin(np.pi * np.arange(1, 5) / 22))

    out['formant_freq'], out['formant_bw'] = _formants(power, freqs, sr)
    return out


def _formants(power: np.ndarray, freqs: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """
    F1-F3 and bandwidths in Hz from an order-11 LPC fit after band-limiting to 0-5.5 kHz.

    The power spectrum up to 5.5 kHz is taken as that of a signal sampled at
    11 kHz (resampling in the spectral domain, as openSMILE does), its inverse
    FFT gives the autocorrelation, and Levinson-Durbin the predictor. Roots
    with 50 < f < 5450 Hz and a bandwidth below 1 kHz are formant
    candidates, f = angle * fs / 2 pi and bandwidth = -fs / pi * ln|z|
    (Markel and Gray, 1976). openSMILE applies no bandwidth limit; without it
    broad poles from the spectral tilt are taken for formants. 0 where a
    frame has fewer than three candidates.
    """
    order = 11
    keep = freqs <= 5500 if sr / 2 > 5500 else np.ones(len(freqs), dtype=bool)
    n_bins = int(keep.sum())
    fs = 2 * (n_bins - 1) * (freqs[1] - freqs[0])
    r = np.fft.irfft(power[:, keep], n=2 * (n_bins - 1), axis=1)[:, :order + 1]
    a, err = levinson(r, order)
    roots = lpc_roots(a)
    f = np.angle(roots) * fs / (2 * np.pi)
    with np.errstate(divide='ignore'):
        bw = -fs / np.pi * np.log(np.abs(roots))
    valid = (roots.imag > 0) & (f > 50) & (f < 5450) & (bw < _FORMANT_MAX_BANDWIDTH) & (err[:, None] > 0)
    f = np.where(valid, f, np.inf)
    order_idx = np.argsort(f, axis=1)[:, :3]
    f = np.take_along_axis(f, order_idx, 1)
    bw = np.take_along_axis(bw, order_idx, 1)
    found = np.isfinite(f)
    return np.where(found, f, 0.0).T, np.where(found, bw, 0.0).T


def _shs(mag: np.ndarray, freqs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    F0 candidates in Hz, their SHS values, and the SHS peak-to-mean ratio per frame.

    Hermes (1988), "Measurement of pitch by subharmonic summation", JASA
    83(1). The magnitude spectrum is resampled on a log-frequency axis (48
    points per octave from 25 Hz, cubic spline), reduced to its peaks and
    their neighbours, smoothed, weighted by Hermes' arctan auditory
    sensitivity curve, and summed over 15 compressed subharmonics
    H(s) = sum_n 0.85^(n-1) P(s + log2 n). The candidates are the six
    largest local maxima of H in 55-1000 Hz, refined by a parabola, best
    first (0 where a frame has fewer).
    """
    s = np.arange(np.log2(25.0), np.log2(freqs[-1]), 1.0 / _SHS_POINTS_PER_OCTAVE)
    spec = np.maximum(CubicSpline(freqs, mag, axis=1)(2.0 ** s), 0.0)
    peak = np.zeros(spec.shape, dtype=bool)
    peak[:, 1:-1] = (spec[:, 1:-1] > spec[:, :-2]) & (spec[:, 1:-1] >= spec[:, 2:])
    keep = peak.copy()
    keep[:, 1:] |= peak[:, :-1]
    keep[:, :-1] |= peak[:, 1:]
    spec = np.pad(np.where(keep, spec, 0.0), ((0, 0), (1, 1)))
    spec = 0.25 * spec[:, :-2] + 0.5 * spec[:, 1:-1] + 0.25 * spec[:, 2:]
    spec *= 0.5 + np.arctan(3 * (s - np.log2(65.0))) / np.pi
    shs = np.zeros_like(spec)
    for n in range(1, _SHS_HARMONICS + 1):
        shift = int(round(np.log2(n) * _SHS_POINTS_PER_OCTAVE))
        if shift < len(s):
            shs[:, :len(s) - shift] += _SHS_COMPRESSION ** (n - 1) * spec[:, shift:]
    band = np.flatnonzero((2.0 ** s >= _F0_MIN) & (2.0 ** s <= _F0_MAX))
    sub = shs[:, band[0] - 1:band[-1] + 2]
    is_max = (sub[:, 1:-1] > sub[:, :-2]) & (sub[:, 1:-1] >= sub[:, 2:])
    values = np.where(is_max, sub[:, 1:-1], -np.inf)
    idx = np.argsort(-values, axis=1)[:, :_SHS_CANDIDATES]
    rows = np.arange(len(sub))[:, None]
    left, mid, right = sub[rows, idx], sub[rows, idx + 1], sub[rows, idx + 2]
    curve = left - 2 * mid + right
    offset = np.where(curve < 0, 0.5 * (left - right) / np.where(curve < 0, curve, -1.0), 0.0)
    found = np.isfinite(np.take_along_axis(values, idx, 1))
    cand_f = np.where(found, 2.0 ** (s[band][idx] + offset / _SHS_POINTS_PER_OCTAVE), 0.0)
    cand_v = np.where(found, mid, 0.0)
    ratio = cand_v[:, 0] / np.maximum(shs.mean(axis=1), _TINY)
    return cand_f, cand_v, ratio


def _track(cand_f: np.ndarray, cand_v: np.ndarray, voiced: np.ndarray) -> np.ndarray:
    """
    F0 path through the SHS candidates of each voiced stretch by dynamic programming.

    The cost of candidate c in a frame is -ln(H_c / H_best) plus 3 per
    octave of change from the previous frame's choice, so the path keeps
    the strongest candidates unless that needs an octave jump.
    """
    f0 = np.zeros(len(voiced))
    edges = np.flatnonzero(np.diff(np.concatenate([[0], voiced.astype(np.int8), [0]])))
    with np.errstate(divide='ignore'):
        local = -np.log(cand_v / np.maximum(cand_v[:, :1], _TINY))
        log_f = np.log2(np.where(cand_f > 0, cand_f, 1.0))
    local[cand_f <= 0] = np.inf
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        cost = local[start].copy()
        back = np.zeros((stop - start, cand_f.shape[1]), dtype=int)
        for t in range(start + 1, stop):
            trans = cost[None, :] + 3.0 * np.abs(log_f[t][:, None] - log_f[t - 1][None, :])
            back[t - start] = np.argmin(trans, axis=1)
            cost = trans[np.arange(len(cost)), back[t - start]] + local[t]
        c = int(np.argmin(cost))
        for t in range(stop - 1, start - 1, -1):
            f0[t] = cand_f[t, c]
            c = back[t - start, c]
    return f0


def _nccf_at(frames: np.ndarray, lag_lo: np.ndarray, lag_hi: np.ndarray) -> np.ndarray:
    """Maximum normalized cross-correlation of x[j] and x[j + tau] over lag_lo <= tau <= lag_hi, per frame."""
    n = frames.shape[1]
    max_lag = int(min(lag_hi.max(), n - 2))
    nfft = _next_pow2(n + max_lag + 1)
    spec = np.fft.rfft(frames, nfft)
    acf = np.fft.irfft(spec.real ** 2 + spec.imag ** 2, nfft)[:, :max_lag + 1]
    lags = np.arange(max_lag + 1)
    cum = np.concatenate([np.zeros((len(frames), 1)), np.cumsum(frames ** 2, axis=1)], axis=1)
    e_lead = cum[:, n - lags]
    e_trail = cum[:, n:] - cum[:, lags]
    r = acf / np.sqrt(np.maximum(e_lead * e_trail, _TINY))
    window = (lags >= lag_lo[:, None]) & (lags <= np.minimum(lag_hi, max_lag)[:, None])
    return np.where(window, r, -np.inf).max(axis=1)


def _clean_runs(voiced: np.ndarray, max_gap: int, min_len: int) -> np.ndarray:
    """Bridge unvoiced gaps of up to max_gap frames inside voicing, then drop voiced runs shorter than min_len."""
    v = voiced.copy()
    edges = np.flatnonzero(np.diff(np.concatenate([[1], v.astype(np.int8), [1]])))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if 0 < start and stop < len(v) and stop - start <= max_gap:
            v[start:stop] = True
    edges = np.flatnonzero(np.diff(np.concatenate([[0], v.astype(np.int8), [0]])))
    for start, stop in zip(edges[::2], edges[1::2], strict=True):
        if stop - start < min_len:
            v[start:stop] = False
    return v


def _pitch(x: np.ndarray, sr: float, n_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """F0 in Hz (0 = unvoiced), the 60 ms Gaussian-window magnitude spectra, and their frequencies."""
    hop = int(round(_HOP * sr))
    length = int(round(_LONG * sr))
    nfft = _next_pow2(length)
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    if n_frames == 0:
        return np.zeros(0), np.zeros((0, len(freqs))), freqs
    raw = _frames(x, length, hop, n_frames)
    window = _gauss(length)
    mag = np.empty((n_frames, len(freqs)))
    cand_f = np.empty((n_frames, _SHS_CANDIDATES))
    cand_v = np.empty((n_frames, _SHS_CANDIDATES))
    ratio, nccf, rms = np.empty(n_frames), np.empty(n_frames), np.empty(n_frames)
    # Blocks of frames bound the memory of the spline and autocorrelation arrays on long recordings
    for b in range(0, n_frames, _BLOCK):
        block = slice(b, b + _BLOCK)
        windowed = raw[block] * window
        mag[block] = np.abs(np.fft.rfft(windowed, nfft))
        cand_f[block], cand_v[block], ratio[block] = _shs(mag[block], freqs)
        period = sr / np.where(cand_f[block, 0] > 0, cand_f[block, 0], _F0_MAX)
        nccf[block] = _nccf_at(raw[block], np.floor(0.9 * period).astype(int), np.ceil(1.1 * period).astype(int))
        rms[block] = np.sqrt(np.mean(windowed ** 2, axis=1))
    voiced = (rms >= _VOICING_RMS) & (nccf > _VOICING_NCCF) & (ratio > _VOICING_SHS) & (cand_f[:, 0] > 0)
    voiced = _clean_runs(voiced, _MAX_GAP_FRAMES, _MIN_VOICED_FRAMES)
    return _refine(_track(cand_f, cand_v, voiced), mag, freqs), mag, freqs


def _refine(f0: np.ndarray, mag: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """
    F0 from the peaks of the first five harmonics near k * f0, weighted by their power.

    The Gaussian window's main lobe is a Gaussian, so a parabola through the
    log magnitudes of the three bins around a peak locates a stationary
    harmonic exactly. Each harmonic gives f_k / k; the SHS grid alone is
    1/48 octave coarse (about 0.25 semitones).
    """
    df = freqs[1] - freqs[0]
    logmag = np.log(np.maximum(mag, _TINY))
    out = f0.copy()
    for i in np.flatnonzero(f0 > 0):
        num = den = 0.0
        for k in range(1, 6):
            lo = max(1, int(np.ceil((k - 0.25) * f0[i] / df)))
            hi = min(len(freqs) - 2, int(np.floor((k + 0.25) * f0[i] / df)))
            if hi < lo:
                break
            j = lo + int(np.argmax(mag[i, lo:hi + 1]))
            a, b, c = logmag[i, j - 1:j + 2]
            curve = a - 2 * b + c
            if b < a or b < c or curve >= 0:
                continue
            weight = mag[i, j] ** 2
            num += weight * (j + 0.5 * (a - c) / curve) * df / k
            den += weight
        if den > 0:
            out[i] = num / den
    return out


def _periods(x: np.ndarray, sr: float, f0: np.ndarray, hop: int) -> tuple[np.ndarray, ...]:
    """
    Glottal periods found by waveform matching within voiced stretches.

    Starting at the largest sample of the first expected period, each next
    period length is the lag within +-10 % of the F0 period that maximizes
    the normalized cross-correlation of consecutive one-period segments,
    refined by a parabola. A match below 0.5 ends the chain. Returns period
    start positions (samples), lengths (samples), peak-to-peak amplitudes
    and chain ids.
    """
    starts, lengths, amps, chains = [], [], [], []
    edges = np.flatnonzero(np.diff(np.concatenate([[0], (f0 > 0).astype(np.int8), [0]])))
    chain = 0
    for first, stop in zip(edges[::2], edges[1::2], strict=True):
        end = min(len(x), (stop - 1) * hop + int(round(_LONG * sr)))
        centres = np.arange(first, stop) * hop + 0.5 * _LONG * sr
        p = float(first * hop)
        new_chain = True
        while True:
            t0 = sr / np.interp(p, centres, f0[first:stop])
            seg = int(round(t0))
            lo, hi = int(np.floor(0.9 * t0)), int(np.ceil(1.1 * t0))
            if new_chain:
                if p + seg + hi + seg > end:
                    break
                p = float(int(p) + np.argmax(np.abs(x[int(p):int(p) + seg])))
                new_chain = False
            i = int(round(p))
            if i + hi + seg > end:
                break
            ref = x[i:i + seg]
            cand = np.lib.stride_tricks.sliding_window_view(x[i + lo:i + hi + seg], seg)
            num = cand @ ref
            den = np.sqrt(np.maximum((ref @ ref) * np.einsum('ij,ij->i', cand, cand), _TINY))
            cc = num / den
            k = int(np.argmax(cc))
            if cc[k] < 0.5:
                chain += 1
                new_chain = True
                p += t0
                continue
            shift = 0.0
            if 0 < k < len(cc) - 1:
                curve = cc[k - 1] - 2 * cc[k] + cc[k + 1]
                if curve < 0:
                    shift = 0.5 * (cc[k - 1] - cc[k + 1]) / curve
            length = lo + k + shift
            starts.append(p)
            lengths.append(length)
            period = x[i:i + int(round(length))]
            amps.append(period.max() - period.min())
            chains.append(chain)
            p += length
        chain += 1
    return np.array(starts), np.array(lengths), np.array(amps), np.array(chains)


def _jitter_shimmer(x: np.ndarray, sr: float, f0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Local jitter and shimmer (dB) per 60 ms frame from the periods starting in it.

    jitter = mean |T_i - T_{i-1}| / mean T_i and shimmer = mean |20 log10(A_i / A_{i-1})|
    over consecutive periods of one chain (Eyben et al., 2016); 0 where a
    frame holds fewer than two such periods.
    """
    hop = int(round(_HOP * sr))
    jitter = np.zeros(len(f0))
    shimmer = np.zeros(len(f0))
    starts, lengths, amps, chains = _periods(x, sr, f0, hop)
    if len(starts) < 2:
        return jitter, shimmer
    pair = chains[1:] == chains[:-1]
    d_len = np.where(pair, np.abs(np.diff(lengths)), 0.0)
    ratio = np.where(pair & (amps[1:] > 0) & (amps[:-1] > 0), amps[1:] / np.where(amps[:-1] > 0, amps[:-1], 1), 1.0)
    d_amp = np.abs(20 * np.log10(np.maximum(ratio, _TINY)))
    pair_len = np.where(pair, lengths[1:], 0.0)
    cums = [np.concatenate([[0.0], np.cumsum(v)]) for v in (d_len, d_amp, pair_len, pair.astype(float))]
    frame_lo = np.arange(len(f0)) * hop
    a = np.searchsorted(starts[1:], frame_lo)
    b = np.searchsorted(starts[1:], frame_lo + int(round(_LONG * sr)))
    n = cums[3][b] - cums[3][a]
    ok = (n >= 1) & (f0 > 0)
    mean_len = (cums[2][b] - cums[2][a]) / np.maximum(n, 1)
    jitter[ok] = ((cums[0][b] - cums[0][a]) / np.maximum(n, 1) / np.maximum(mean_len, _TINY))[ok]
    shimmer[ok] = ((cums[1][b] - cums[1][a]) / np.maximum(n, 1))[ok]
    return jitter, shimmer


def _harmonic_llds(mag: np.ndarray, freqs: np.ndarray, sr: float, f0: np.ndarray,
                   formants: np.ndarray) -> dict[str, np.ndarray]:
    """
    HNR, H1-H2, H1-A3 and formant levels relative to H1, in dB, from the 60 ms spectra.

    HNR = 10 log10(r / (1 - r)), r the largest autocorrelation within +-10 %
    of the F0 period over the zero-lag value, the autocorrelation taken from
    the Gaussian-windowed power spectrum. Harmonic level h(k) is the largest
    magnitude within k F0 +- F0 / 2 in dB. Following openSMILE, 'H1-H2' is
    h(2) - h(3) and 'H1-A3' is h(2) - A3, while the formant levels are
    h(k_i) - h(1) with k_i the harmonic closest to formant i; A3 = h(k_3).
    All are 0 in unvoiced frames.
    """
    n_frames = len(f0)
    out = {k: np.zeros(n_frames) for k in ('hnr', 'h1h2', 'h1a3')}
    out['formant_amp'] = np.zeros((3, n_frames))
    if n_frames == 0:
        return out
    nfft = 2 * (mag.shape[1] - 1)
    acf = np.fft.irfft(mag ** 2, nfft)
    df = freqs[1] - freqs[0]
    for i in np.flatnonzero(f0 > 0):
        period = sr / f0[i]
        lo, hi = int(0.9 * period), min(int(np.ceil(1.1 * period)), nfft // 2)
        if acf[i, 0] <= 0 or lo >= hi:
            continue
        r = np.clip(acf[i, lo:hi + 1].max() / acf[i, 0], 1e-10, 1 - 1e-10)
        out['hnr'][i] = 10 * np.log10(r / (1 - r))
        h1, h2, h3 = (_level(mag[i], f0[i] / df, k) for k in (1, 2, 3))
        if h2 is not None and h3 is not None:
            out['h1h2'][i] = h2 - h3
        for j in range(3):
            if formants[j, i] <= 0 or h1 is None:
                continue
            a_j = _level(mag[i], f0[i] / df, max(1, int(round(formants[j, i] / f0[i]))))
            if a_j is None:
                continue
            out['formant_amp'][j, i] = a_j - h1
            if j == 2 and h2 is not None:
                out['h1a3'][i] = h2 - a_j
    return out


def _level(mag: np.ndarray, f0_bins: float, k: int) -> float | None:
    """Largest magnitude within (k +- 1/2) F0 in dB, None beyond Nyquist; F0 given in FFT bins."""
    a = int(np.ceil((k - 0.5) * f0_bins))
    b = int(np.floor((k + 0.5) * f0_bins))
    if b >= len(mag) or b < a:
        return None
    return 20 * np.log10(max(mag[a:b + 1].max(), _TINY))


def _sma3(v: np.ndarray, nonzero: bool = False) -> np.ndarray:
    """
    Three-frame moving average along the last axis, repeating the end frames.

    With ``nonzero`` (openSMILE's 'sma3nz') zeros mark undefined frames:
    they stay 0 and are left out of their neighbours' averages.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] == 0:
        return v.copy()
    weight = (v != 0).astype(np.float64) if nonzero else np.ones_like(v)
    pad = [(0, 0)] * (v.ndim - 1) + [(1, 1)]
    vp, wp = np.pad(v * weight, pad, mode='edge'), np.pad(weight, pad, mode='edge')
    total = vp[..., :-2] + vp[..., 1:-1] + vp[..., 2:]
    count = wp[..., :-2] + wp[..., 1:-1] + wp[..., 2:]
    out = total / np.maximum(count, 1)
    return np.where(weight > 0, out, 0.0) if nonzero else out


def _lld(x: np.ndarray, sr: float) -> dict[str, Any]:
    hop = int(round(_HOP * sr))
    short = int(round(_SHORT * sr))
    long_ = int(round(_LONG * sr))
    n_short = 1 + (len(x) - short) // hop if len(x) >= short else 1
    n_long = 1 + (len(x) - long_) // hop if len(x) >= long_ else 0
    spectral = _spectral_llds(x, sr, n_short)
    f0, mag, freqs = _pitch(x, sr, n_long)
    voiced = np.zeros(n_short, dtype=bool)
    voiced[:n_long] = f0 > 0
    formants = np.where(voiced[:n_long], spectral['formant_freq'][:, :n_long], 0.0)
    bandwidths = np.where(voiced[:n_long], spectral['formant_bw'][:, :n_long], 0.0)
    jitter, shimmer = _jitter_shimmer(x, sr, f0)
    harmonic = _harmonic_llds(mag, freqs, sr, f0, formants)
    semitone = np.where(f0 > 0, 12 * np.log2(np.where(f0 > 0, f0, 1) / 27.5), 0.0)
    rows_f = [semitone, jitter, shimmer, harmonic['hnr'], harmonic['h1h2'], harmonic['h1a3']]
    for j in range(3):
        rows_f += [formants[j], bandwidths[j], harmonic['formant_amp'][j]]
    rows_e = [spectral['loudness'], spectral['alpha'], spectral['hammarberg'], spectral['slope_low'],
              spectral['slope_high'], spectral['flux'], *spectral['mfcc'].T]
    return {
        'e': np.array(rows_e),
        'f': np.array(rows_f).reshape(15, n_long),
        'voiced': voiced,
        'n_long': n_long,
        'power': spectral['power'],
        'hop': hop,
    }


def _check(signal, sr) -> np.ndarray:
    signal = validate_signal(signal).astype(np.float64, copy=False)
    validate_sr(sr)
    if sr < 8000:
        raise ValueError(f'eGeMAPS needs a sample rate of at least 8000 Hz, got {sr}. Resample the signal first.')
    return signal


def extract_egemaps_lld(signal: np.ndarray, sr: int, *, return_metadata: bool = False) -> dict[str, Any]:
    """
    The 25 eGeMAPS v02 low-level descriptors per 10 ms frame, shape (25, n_frames).

    Rows follow EGEMAPS_LLD_NAMES (openSMILE's column names): loudness,
    alpha ratio, Hammarberg index, spectral slopes 0-500 and 500-1500 Hz,
    spectral flux and MFCC 1-4 on every frame (3-frame smoothed, 'sma3'),
    then F0 (semitones above 27.5 Hz), jitter, shimmer (dB), HNR (dB),
    H1-H2, H1-A3 and F1-F3 frequency, bandwidth (Hz) and level relative to
    H1 (dB), smoothed over voiced neighbours ('sma3nz') and NaN where
    undefined (unvoiced frames, too few periods for jitter and shimmer).
    Frame i starts at i * 10 ms; n_frames = 1 + (len - 20 ms) // 10 ms.
    metadata: 'times' (centre of each 20 ms frame, s) and 'names'.
    """
    x = _check(signal, sr)
    lld = _lld(x, sr)
    n = lld['e'].shape[1]
    f = np.zeros((15, n))
    f[:, :lld['n_long']] = _sma3(lld['f'], nonzero=True)
    values = np.vstack([_sma3(lld['e']), np.where(f != 0, f, np.nan)])
    out: dict[str, Any] = {'egemaps_lld': values}
    if return_metadata:
        times = (np.arange(n) * lld['hop'] + 0.5 * _SHORT * sr) / float(sr)
        out['metadata'] = {'times': times, 'names': list(EGEMAPS_LLD_NAMES)}
    return out


def _mean_cv(v: np.ndarray) -> tuple[float, float]:
    if v.size == 0:
        return 0.0, 0.0
    mean = float(v.mean())
    return mean, float(v.std() / mean) if mean != 0 else 0.0


def _slopes(contour: np.ndarray, dt: float) -> tuple[float, float, float, float]:
    """
    Mean and std of rising and falling slopes (units per second) between turning points.

    Peaks are found by _peaks; the minimum between two peaks is the lowest
    point between them. Rising slopes run from each minimum to the next peak, falling
    slopes from each peak to the next minimum and are reported as positive
    rates, except a final fall that reaches the end of the contour, which
    keeps its negative sign as openSMILE reports it. A rise after the last
    minimum to the end of the contour counts as a rising slope.
    """
    peaks = _peaks(contour)
    if not peaks:
        return 0.0, 0.0, 0.0, 0.0
    rising, falling = [], []
    bounds = [0, *peaks, len(contour) - 1]
    for k, p in enumerate(peaks):
        before = bounds[k] + int(np.argmin(contour[bounds[k]:p + 1]))
        after = p + int(np.argmin(contour[p:bounds[k + 2] + 1]))
        if before < p:
            rising.append((contour[p] - contour[before]) / ((p - before) * dt))
        if after > p:
            fall = (contour[p] - contour[after]) / ((after - p) * dt)
            falling.append(-fall if after == len(contour) - 1 else fall)
    last = peaks[-1] + int(np.argmin(contour[peaks[-1]:]))
    if last < len(contour) - 1:
        rising.append((contour[-1] - contour[last]) / ((len(contour) - 1 - last) * dt))
    r, f = np.array(rising), np.array(falling)
    return (float(r.mean()) if r.size else 0.0, float(r.std()) if r.size else 0.0,
            float(f.mean()) if f.size else 0.0, float(f.std()) if f.size else 0.0)


def _peaks(contour: np.ndarray) -> list[int]:
    """
    Indices of maxima reached by a rise of more than 10 % of the contour's range (hysteresis).

    A maximum counts once the contour falls by that much again, or, for the
    last one, when it is not the final frame.
    """
    if contour.size < 3:
        return []
    thresh = 0.1 * (contour.max() - contour.min())
    if thresh <= 0:
        return []
    peaks = []
    low = high = 0
    state = 0  # 1 after a confirmed rise, -1 after a confirmed fall
    for i in range(1, len(contour)):
        if contour[i] > contour[high]:
            high = i
        if contour[i] < contour[low]:
            low = i
        if state != -1 and contour[high] - contour[i] > thresh:
            if state == 1:
                peaks.append(high)
            state, low = -1, i
        elif state != 1 and contour[i] - contour[low] > thresh:
            state, high = 1, i
    # A last maximum needs the rise but not the confirming fall, unless it is the final frame
    if state == 1 and high < len(contour) - 1:
        peaks.append(high)
    return peaks


def _contour_functionals(v: np.ndarray, dt: float) -> list[float]:
    if v.size == 0:
        return [0.0] * 10
    mean, cv = _mean_cv(v)
    p20, p50, p80 = np.percentile(v, [20, 50, 80])
    return [mean, cv, float(p20), float(p50), float(p80), float(p80 - p20), *_slopes(v, dt)]


def _segments(voiced: np.ndarray, dt: float) -> tuple[list[float], list[float], int]:
    """Voiced and unvoiced segment lengths in seconds, measured first to last frame ((frames - 1) * dt)."""
    edges = np.flatnonzero(np.diff(np.concatenate([[-1], voiced.astype(np.int8), [-1]])))
    runs = np.diff(edges)
    kinds = voiced[edges[:-1]] if len(edges) > 1 else np.zeros(0, dtype=bool)
    lengths = (runs - 1) * dt
    return list(lengths[kinds]), list(lengths[~kinds]), int(kinds.sum())


def _functionals(x: np.ndarray, sr: float) -> np.ndarray:
    lld = _lld(x, sr)
    dt = _HOP
    e = _sma3(lld['e'])
    f = _sma3(lld['f'], nonzero=True)
    voiced = lld['voiced']
    values: list[float] = []

    f0 = f[0]
    values += _contour_functionals(f0[f0 != 0], dt)
    values += _contour_functionals(e[0], dt)
    for row in range(5, 10):
        values += list(_mean_cv(e[row]))
    for row in range(1, 15):
        values += list(_mean_cv(f[row][f[row] != 0]))
    # alpha ratio, Hammarberg index, both slopes, flux and MFCC 1-4 split by voicing
    gated_v = _sma3(np.where(voiced, lld['e'][1:10], 0.0), nonzero=True)
    for row in gated_v:
        values += list(_mean_cv(row[row != 0]))
    gated_u = _sma3(np.where(~voiced, lld['e'][1:6], 0.0), nonzero=True)
    for row in gated_u:
        values.append(_mean_cv(row[row != 0])[0])

    duration_e = e.shape[1] * dt
    values.append(len(_peaks(e[0])) / duration_e)
    n_long = lld['n_long']
    v_len, u_len, n_voiced = _segments(f0 != 0, dt) if n_long else ([], [], 0)
    values.append(n_voiced / (n_long * dt) if n_long else 0.0)
    values += [float(np.mean(v_len)) if v_len else 0.0, float(np.std(v_len)) if v_len else 0.0]
    values += [float(np.mean(u_len)) if u_len else 0.0, float(np.std(u_len)) if u_len else 0.0]
    values.append(10 * np.log10(max(float(np.mean(lld['power'])), 1e-12)))
    return np.array(values)


def extract_egemaps(signal: np.ndarray, sr: int, *, return_metadata: bool = False) -> dict[str, Any]:
    """
    The 88 eGeMAPS v02 functionals of the whole clip, shape (88, 1).

    One column per clip, so 'mean' aggregation in a config returns the 88
    values unchanged. Order and names are EGEMAPS_NAMES (openSMILE's).
    Statistics over voiced frames are 0 when the clip has none.
    metadata: 'times' (clip centre, s) and 'names'.
    """
    x = _check(signal, sr)
    out: dict[str, Any] = {'egemaps': _functionals(x, sr)[:, np.newaxis]}
    if return_metadata:
        out['metadata'] = {'times': np.array([0.5 * len(x) / float(sr)]), 'names': list(EGEMAPS_NAMES)}
    return out


def egemaps(signal: np.ndarray, sr: int) -> dict[str, float]:
    """The 88 eGeMAPS v02 functionals as {name: value}, in the order of EGEMAPS_NAMES."""
    return dict(zip(EGEMAPS_NAMES, map(float, _functionals(_check(signal, sr), sr)), strict=True))
