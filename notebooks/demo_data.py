"""
Audio for the example notebooks.

A local copy of ESC-50, UrbanSound8K or Nonspeech7k is used when one of the
environment variables in DATASET_ENV_VARS points at it; otherwise labelled
clips are synthesized with NumPy, so every notebook runs without downloads.
synthetic_voice makes sustained vowels with known F0, jitter and shimmer for
the eGeMAPS notebook.
"""
from __future__ import annotations

import csv
import glob
import os
import random

import numpy as np
import soundfile as sf
from scipy.signal import lfilter

DATASET_ENV_VARS = ('ESC_50_dataset', 'UrbanSound8K_dataset', 'dataset_nonspeech7k')
AUDIO_EXTS = ('.wav', '.ogg', '.flac')
SYNTH_CLASSES = ('tone', 'chirp', 'noise_burst', 'am_fm')

# Metadata column names used by ESC-50, UrbanSound8K and Nonspeech7k
_FILE_COLUMNS = ('filename', 'slice_file_name', 'file_name', 'file')
_LABEL_COLUMNS = ('category', 'class', 'classname', 'label')


def dataset_root() -> tuple[str, str] | None:
    """(env var, root) of the first configured dataset, or None."""
    for var in DATASET_ENV_VARS:
        root = os.environ.get(var)
        if root and os.path.isdir(root):
            return var, root
    return None


def find_audio(max_files: int | None = None) -> list[str]:
    """Sorted audio paths under the configured dataset root (empty without one)."""
    found = dataset_root()
    if found is None:
        return []
    files = sorted(p for ext in AUDIO_EXTS for p in glob.glob(os.path.join(found[1], '**', '*' + ext), recursive=True))
    return files[:max_files] if max_files else files


def labelled_files(n_classes: int = 4, n_per_class: int = 12, seed: int = 0) -> tuple[list[str], list[str]] | None:
    """
    (paths, labels) from the configured dataset's metadata CSV, or None.

    Takes the ``n_classes`` most frequent classes (ties broken by name) and
    ``n_per_class`` random files from each.
    """
    found = dataset_root()
    if found is None:
        return None
    by_name = {os.path.basename(p): p for p in find_audio()}
    buckets: dict[str, list[str]] = {}
    for meta in glob.glob(os.path.join(found[1], '**', '*.csv'), recursive=True):
        with open(meta, newline='', encoding='utf-8', errors='replace') as fh:
            reader = csv.DictReader(fh)
            columns = {c.lower().strip(): c for c in reader.fieldnames or []}
            file_col = next((columns[c] for c in _FILE_COLUMNS if c in columns), None)
            label_col = next((columns[c] for c in _LABEL_COLUMNS if c in columns), None)
            if file_col is None or label_col is None:
                continue
            for row in reader:
                path = by_name.get(os.path.basename(row[file_col]))
                if path is not None:
                    buckets.setdefault(row[label_col], []).append(path)
    if len(buckets) < 2:
        return None
    rng = random.Random(seed)
    classes = sorted(buckets, key=lambda c: (-len(set(buckets[c])), c))[:n_classes]
    paths: list[str] = []
    labels: list[str] = []
    for label in classes:
        chosen = sorted(set(buckets[label]))
        rng.shuffle(chosen)
        paths.extend(chosen[:n_per_class])
        labels.extend([label] * len(chosen[:n_per_class]))
    return paths, labels


def synth_clip(kind: str, seconds: float = 1.0, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """
    One synthetic clip of class ``kind`` (see SYNTH_CLASSES), float32 in [-1, 1].

    Each class draws its parameters at random (fundamental, sweep range,
    burst timing, modulation rates) and adds white noise at 5 to 30 dB SNR,
    so the classes overlap in level and noise and a classifier has to use
    the spectral and temporal shape.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    t = np.arange(n) / sr
    if kind == 'tone':
        f0 = rng.uniform(150, 600)
        x = sum(rng.uniform(0.2, 1.0) / k * np.sin(2 * np.pi * k * f0 * t + rng.uniform(0, 2 * np.pi))
                for k in range(1, int(rng.integers(2, 6)) + 1))
    elif kind == 'chirp':
        f_lo, f_hi = rng.uniform(200, 800), rng.uniform(2000, 6000)
        if rng.random() < 0.5:
            f_lo, f_hi = f_hi, f_lo
        x = np.sin(2 * np.pi * np.cumsum(np.linspace(f_lo, f_hi, n)) / sr)
    elif kind == 'noise_burst':
        env = np.zeros(n)
        for _ in range(int(rng.integers(2, 6))):
            start, length = int(rng.integers(0, n)), int(rng.uniform(0.03, 0.15) * sr)
            env[start:start + length] = rng.uniform(0.5, 1.0)
        x = env * rng.standard_normal(n)
    elif kind == 'am_fm':
        fc, fm, am = rng.uniform(500, 2000), rng.uniform(3, 8), rng.uniform(2, 12)
        x = (1 + 0.8 * np.sin(2 * np.pi * am * t)) * np.sin(
            2 * np.pi * fc * t + rng.uniform(20, 80) / fm * np.sin(2 * np.pi * fm * t))
    else:
        raise ValueError(f'Unknown kind {kind!r}; one of {SYNTH_CLASSES}')
    x = x / (np.max(np.abs(x)) + 1e-12)
    snr_db = rng.uniform(5, 30)
    x = x + rng.standard_normal(n) * np.sqrt(np.mean(x ** 2) / 10 ** (snr_db / 10))
    return (0.8 * x / np.max(np.abs(x))).astype(np.float32)


VOWEL_FORMANTS = {
    'a': ((730, 1090, 2440), (90, 110, 160)),
    'i': ((270, 2290, 3010), (60, 90, 150)),
    'u': ((300, 870, 2240), (60, 90, 150)),
}


def synthetic_voice(
    f0: float | tuple[float, float] = 140.0,
    jitter: float = 0.0,
    shimmer: float = 0.0,
    vowel: str = 'a',
    seconds: float = 2.0,
    voiced: tuple[float, float] = (0.25, 1.75),
    snr_db: float = 40.0,
    sr: int = 16000,
    seed: int = 0,
    return_truth: bool = False,
):
    """
    A sustained vowel: Rosenberg glottal pulses through three formant resonators, float64.

    ``f0`` in Hz is constant, or a (start, end) glide over the voiced
    stretch. Each period is the nominal one times 1 + jitter * N(0, 1) and
    each pulse amplitude 1 + shimmer * N(0, 1). The pulse shape follows
    the nominal period, so the interval between glottal closures equals the
    drawn period. Silence surrounds the voiced stretch; white noise sets
    the SNR over it.

    With ``return_truth`` also returns the realized values in eGeMAPS
    units: mean F0 in semitones above 27.5 Hz, jitter as mean |T_i -
    T_i-1| / mean T, and shimmer as mean |20 log10(A_i / A_i-1)| in dB.
    """
    rng = np.random.default_rng(seed)
    n = int(round(seconds * sr))
    f_start, f_end = (f0, f0) if np.isscalar(f0) else f0
    start, stop = voiced
    flow = np.zeros(n)
    periods, amps = [], []
    pos = start
    while pos < stop:
        nominal = 1 / (f_start + (f_end - f_start) * (pos - start) / (stop - start))
        period = nominal * (1 + jitter * rng.standard_normal())
        amp = max(0.05, 1 + shimmer * rng.standard_normal())
        i0, i1 = int(np.ceil(pos * sr)), min(int(np.ceil((pos + nominal) * sr)), n)
        phase = (np.arange(i0, i1) / sr - pos) / nominal
        # Rosenberg (1971): opening over 60 % of the nominal period, closing over 25 %
        pulse = np.where(phase < 0.6, 0.5 * (1 - np.cos(np.pi * phase / 0.6)),
                         np.where(phase < 0.85, np.cos(np.pi * (phase - 0.6) / 0.5), 0.0))
        flow[i0:i1] += amp * pulse
        periods.append(period)
        amps.append(amp)
        pos += period
    y = np.diff(flow, prepend=0.0)
    for fc, bw in zip(*VOWEL_FORMANTS[vowel], strict=True):
        r = np.exp(-np.pi * bw / sr)
        y = lfilter([1 - r], [1, -2 * r * np.cos(2 * np.pi * fc / sr), r * r], y)
    # Lip radiation, a first difference
    y = np.diff(y, prepend=0.0)
    y *= 0.3 / max(np.max(np.abs(y)), 1e-12)
    power = np.mean(y[int(start * sr):int(stop * sr)] ** 2)
    y = y + rng.standard_normal(n) * np.sqrt(power / 10 ** (snr_db / 10))
    if not return_truth:
        return y
    t, a = np.array(periods), np.array(amps)
    truth = {
        'f0_semitones': float(np.mean(12 * np.log2(1 / t / 27.5))),
        'jitter': float(np.mean(np.abs(np.diff(t))) / np.mean(t)),
        'shimmer_db': float(np.mean(np.abs(20 * np.log10(a[1:] / a[:-1])))),
    }
    return y, truth


def synthetic_dataset(
    n_per_class: int = 12,
    seconds: float = 1.0,
    sr: int = 16000,
    classes: tuple[str, ...] = SYNTH_CLASSES,
    seed: int = 0,
) -> tuple[list[np.ndarray], np.ndarray]:
    """(signals, labels) with ``n_per_class`` synthetic clips per class, interleaved."""
    signals, labels = [], []
    for i in range(n_per_class):
        for c, kind in enumerate(classes):
            signals.append(synth_clip(kind, seconds, sr, seed=seed + 1000 * c + i))
            labels.append(kind)
    return signals, np.array(labels)


def write_synthetic_folder(folder: str, n_per_class: int = 4, seconds: float = 1.0, sr: int = 16000,
                           seed: int = 0) -> tuple[list[str], np.ndarray]:
    """Write synthetic_dataset() as <folder>/<label>/<label>_<i>.wav; returns (paths, labels)."""
    signals, labels = synthetic_dataset(n_per_class, seconds, sr, seed=seed)
    paths = []
    counts: dict[str, int] = {}
    for x, label in zip(signals, labels, strict=True):
        i = counts[label] = counts.get(label, -1) + 1
        os.makedirs(os.path.join(folder, label), exist_ok=True)
        path = os.path.join(folder, label, f'{label}_{i:02d}.wav')
        sf.write(path, x, sr)
        paths.append(path)
    return paths, labels
