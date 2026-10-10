"""
Audio for the example notebooks.

A local copy of ESC-50, UrbanSound8K or Nonspeech7k is used when one of the
environment variables in DATASET_ENV_VARS points at it; otherwise labelled
clips are synthesized with NumPy, so every notebook runs without downloads.
"""
from __future__ import annotations

import csv
import glob
import os
import random

import numpy as np
import soundfile as sf

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
