"""
Synthetic voice for the eGeMAPS tests and benchmarks/validate_egemaps.py.

A Rosenberg glottal pulse train with a controlled F0 contour, cycle-to-cycle
period jitter and amplitude shimmer, filtered by a cascade of second-order
formant resonators, with lip radiation, fricative noise between voiced
stretches and white noise at a given SNR.
"""
import numpy as np
from scipy.signal import butter, lfilter

VOWELS = {
    'a': ((730, 1090, 2440), (90, 110, 160)),
    'e': ((530, 1840, 2480), (60, 100, 150)),
    'i': ((270, 2290, 3010), (60, 90, 150)),
    'o': ((570, 840, 2410), (70, 80, 150)),
    'u': ((300, 870, 2240), (60, 90, 150)),
}


def rosenberg(phase: np.ndarray, open_quotient: float = 0.6, closing_quotient: float = 0.25) -> np.ndarray:
    """Rosenberg (1971) glottal flow pulse for phase in [0, 1)."""
    flow = np.zeros_like(phase)
    rise = phase < open_quotient
    flow[rise] = 0.5 * (1 - np.cos(np.pi * phase[rise] / open_quotient))
    fall = (phase >= open_quotient) & (phase < open_quotient + closing_quotient)
    flow[fall] = np.cos(np.pi * (phase[fall] - open_quotient) / (2 * closing_quotient))
    return flow


def synthetic_voice(
    sr: int = 16000,
    duration: float = 3.0,
    f0: tuple[float, float] = (120.0, 180.0),
    contour: str = 'arch',
    jitter: float = 0.005,
    shimmer: float = 0.03,
    formants: tuple[float, ...] = (730, 1090, 2440),
    bandwidths: tuple[float, ...] = (90, 110, 160),
    snr_db: float = 30.0,
    voiced: tuple[tuple[float, float], ...] = ((0.2, 1.3), (1.6, 2.7)),
    fricative_db: float | None = -25.0,
    level: float = 0.3,
    seed: int = 0,
) -> np.ndarray:
    """
    One synthetic utterance, peak ``level``, as float64.

    ``f0`` gives the start and peak (arch), start and end (glide) or centre
    and depth in Hz (vibrato, 5 Hz rate) of the contour in each voiced
    stretch. Each period is the nominal one times (1 + jitter * N(0, 1)),
    each pulse amplitude 1 + shimmer * N(0, 1). ``fricative_db`` adds
    high-passed noise at that level relative to the voice peak in the
    unvoiced gaps (None for silence). White noise sets the SNR over the
    voiced samples.
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration * sr))
    t = np.arange(n) / sr
    flow = np.zeros(n)
    for start, stop in voiced:
        pos = start
        while pos < stop:
            frac = (pos - start) / (stop - start)
            if contour == 'arch':
                nominal = f0[0] + (f0[1] - f0[0]) * np.sin(np.pi * frac)
            elif contour == 'glide':
                nominal = f0[0] + (f0[1] - f0[0]) * frac
            elif contour == 'vibrato':
                nominal = f0[0] + f0[1] * np.sin(2 * np.pi * 5 * (pos - start))
            else:
                raise ValueError(f'unknown contour {contour!r}')
            period = (1 + jitter * rng.standard_normal()) / nominal
            amp = max(0.05, 1 + shimmer * rng.standard_normal())
            i0 = int(np.ceil(pos * sr))
            i1 = min(int(np.ceil((pos + period) * sr)), n)
            envelope = np.sin(np.pi * min(frac, 1.0)) ** 0.3
            flow[i0:i1] += amp * envelope * rosenberg((t[i0:i1] - pos) / period)
            pos += period
    y = np.diff(flow, prepend=0.0)
    for fc, bw in zip(formants, bandwidths, strict=True):
        r = np.exp(-np.pi * bw / sr)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / sr), r * r], y)
    y = np.diff(y, prepend=0.0)
    y *= level / max(np.max(np.abs(y)), 1e-12)
    is_voiced = np.zeros(n, dtype=bool)
    for start, stop in voiced:
        is_voiced[int(start * sr):int(stop * sr)] = True
    if fricative_db is not None:
        b, a = butter(4, min(3000.0, 0.4 * sr) / (sr / 2), 'high')
        hiss = lfilter(b, a, rng.standard_normal(n))
        hiss *= level * 10 ** (fricative_db / 20) / max(np.max(np.abs(hiss)), 1e-12)
        y += np.where(is_voiced, 0.0, hiss)
    signal_power = np.mean(y[is_voiced] ** 2) if is_voiced.any() else level ** 2
    y += rng.standard_normal(n) * np.sqrt(signal_power / 10 ** (snr_db / 10))
    return y


def corpus(n: int = 20, sr: int = 16000, seed: int = 0) -> list[tuple[str, np.ndarray, dict]]:
    """``n`` (name, signal, parameters) with randomly drawn F0, contour, jitter, shimmer, vowel, SNR, timing, level."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(n):
        vowel = 'aeiou'[k % 5]
        freqs, bws = VOWELS[vowel]
        scale = rng.uniform(0.95, 1.2)
        duration = rng.uniform(3.0, 5.0)
        n_seg = int(rng.integers(1, 4))
        cuts = np.sort(rng.uniform(0.15, duration - 0.15, 2 * n_seg))
        segments = tuple((a, b) for a, b in zip(cuts[::2], cuts[1::2], strict=True) if b - a > 0.15)
        contour = ('arch', 'glide', 'vibrato')[k % 3]
        base = rng.uniform(90, 240)
        f0 = (base, rng.uniform(5, 25)) if contour == 'vibrato' else (base, base * rng.uniform(0.7, 1.6))
        params = {
            'f0': f0, 'contour': contour,
            'jitter': rng.uniform(0.0, 0.03), 'shimmer': rng.uniform(0.0, 0.15),
            'formants': tuple(f * scale for f in freqs), 'bandwidths': bws,
            'snr_db': rng.uniform(5, 40), 'voiced': segments or ((0.3, duration - 0.3),),
            'fricative_db': None if k % 4 == 3 else rng.uniform(-35, -15),
            'level': 10 ** (rng.uniform(-30, -6) / 20),
        }
        name = f'{k:02d}_{vowel}_{contour}_f{base:.0f}_j{params["jitter"]:.3f}_s{params["shimmer"]:.2f}'
        out.append((name, synthetic_voice(sr=sr, duration=duration, seed=seed * 1000 + k, **params), params))
    return out
