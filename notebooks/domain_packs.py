"""
Helpers for the domain presets notebook (07_domain_presets.ipynb).

Each supported domain maps a curated AFX preset (``afx_audio.presets``) to a small,
openly-licensed dataset and a tiny end-to-end classification task:

    ============  =========================  ==============================
    Domain        Dataset                    Task
    ============  =========================  ==============================
    speech        RAVDESS (speech audio)     emotion classification
    bioacoustics  ESC-50 (animal subset)     animal-sound classification
    music         GTZAN                      music-genre classification
    machinery     DCASE2020 T2 dev (pump)    normal vs. anomaly detection
    ============  =========================  ==============================

The datasets are downloaded on demand into ``$DATASETS``. Without ``DATASETS``,
with ``AFX_SYNTHETIC=1``, or when a download fails, the notebook runs on
synthetic stand-in clips from ``demo_data`` instead.
"""
from __future__ import annotations

import csv
import os
import random
import sys
import tarfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from demo_data import write_synthetic_folder

from afx_audio import extract_all_features, flatten_features, load_audio


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
def datasets_root() -> str | None:
    """The download root from the ``DATASETS`` environment variable, or None if unset."""
    root = os.environ.get('DATASETS')
    if not root:
        return None
    os.makedirs(root, exist_ok=True)
    return root


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = 100.0 * done / total_size
    sys.stdout.write(f'\r    downloading… {pct:5.1f}%  '
                     f'({done/1e6:7.1f} / {total_size/1e6:7.1f} MB)')
    sys.stdout.flush()
    if done >= total_size:
        sys.stdout.write('\n')


def download_file(url: str, dest: str) -> str:
    """Download ``url`` to ``dest`` (skipped if the file already exists)."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f'    already downloaded: {os.path.basename(dest)}')
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    print(f'    GET {url}')
    urllib.request.urlretrieve(url, tmp, _reporthook)
    os.replace(tmp, dest)
    return dest


def _extract(archive: str, dest_dir: str) -> None:
    os.makedirs(dest_dir, exist_ok=True)
    if archive.endswith('.zip'):
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    elif archive.endswith(('.tar.gz', '.tgz', '.tar')):
        mode = 'r:gz' if archive.endswith(('.tar.gz', '.tgz')) else 'r:'
        with tarfile.open(archive, mode) as t:
            t.extractall(dest_dir)
    else:
        raise ValueError(f'Unknown archive type: {archive}')


# --------------------------------------------------------------------------- #
# Domain registry
# --------------------------------------------------------------------------- #
@dataclass
class DomainSpec:
    """Everything the notebook needs to run one domain end-to-end."""
    name: str
    preset: str
    title: str
    task: str
    dataset_name: str
    dataset_url: str
    #: Prepare the dataset and return the directory that holds the audio.
    prepare: Callable[[str], str]
    #: (data_dir, classes, n_per_class, seed) -> list of (path, label)
    collect: Callable[..., list[tuple[str, str]]]
    #: Default class labels to keep (None = discover from data).
    default_classes: Sequence[str] | None = None
    approx_download_mb: int = 0


# ---- speech: RAVDESS -------------------------------------------------------- #
_RAVDESS_URL = ('https://zenodo.org/records/1188976/files/'
                'Audio_Speech_Actors_01-24.zip')
_RAVDESS_EMOTION = {
    '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
    '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised',
}


def _prepare_ravdess(root: str) -> str:
    d = os.path.join(root, 'ravdess')
    marker = os.path.join(d, 'Actor_01')
    if not os.path.isdir(marker):
        arc = download_file(_RAVDESS_URL, os.path.join(root, 'ravdess.zip'))
        print('    extracting…')
        _extract(arc, d)
    return d


def _collect_ravdess(data_dir, classes, n_per_class, seed):
    buckets: dict[str, list[str]] = {}
    for r, _, files in os.walk(data_dir):
        for f in files:
            if not f.endswith('.wav'):
                continue
            parts = f.split('-')
            if len(parts) < 3:
                continue
            emo = _RAVDESS_EMOTION.get(parts[2])
            if emo is None or (classes and emo not in classes):
                continue
            buckets.setdefault(emo, []).append(os.path.join(r, f))
    return _balance(buckets, n_per_class, seed)


# ---- bioacoustics: ESC-50 --------------------------------------------------- #
_ESC50_URL = 'https://github.com/karolpiczak/ESC-50/archive/master.zip'
_ESC50_ANIMALS = ['dog', 'rooster', 'pig', 'cow', 'frog',
                  'cat', 'hen', 'insects', 'sheep', 'crow']


def _prepare_esc50(root: str) -> str:
    d = os.path.join(root, 'esc50')
    inner = os.path.join(d, 'ESC-50-master')
    if not os.path.isdir(inner):
        arc = download_file(_ESC50_URL, os.path.join(root, 'esc50.zip'))
        print('    extracting…')
        _extract(arc, d)
    return inner


def _collect_esc50(data_dir, classes, n_per_class, seed):
    meta = os.path.join(data_dir, 'meta', 'esc50.csv')
    audio = os.path.join(data_dir, 'audio')
    buckets: dict[str, list[str]] = {}
    with open(meta, newline='') as fh:
        for row in csv.DictReader(fh):
            cat = row['category']
            if classes and cat not in classes:
                continue
            buckets.setdefault(cat, []).append(
                os.path.join(audio, row['filename']))
    return _balance(buckets, n_per_class, seed)


# ---- music: GTZAN ----------------------------------------------------------- #
_GTZAN_URL = ('https://huggingface.co/datasets/marsyas/gtzan/'
              'resolve/main/data/genres.tar.gz')
_GTZAN_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop',
                 'jazz', 'metal', 'pop', 'reggae', 'rock']


def _prepare_gtzan(root: str) -> str:
    d = os.path.join(root, 'gtzan')
    # The tarball extracts to a "genres"/"genres_original" folder.
    for cand in ('genres', 'genres_original'):
        if os.path.isdir(os.path.join(d, cand)):
            return os.path.join(d, cand)
    arc = download_file(_GTZAN_URL, os.path.join(root, 'gtzan.tar.gz'))
    print('    extracting…')
    _extract(arc, d)
    for cand in ('genres', 'genres_original'):
        if os.path.isdir(os.path.join(d, cand)):
            return os.path.join(d, cand)
    return d


def _collect_gtzan(data_dir, classes, n_per_class, seed):
    buckets: dict[str, list[str]] = {}
    for genre in sorted(os.listdir(data_dir)):
        gdir = os.path.join(data_dir, genre)
        if not os.path.isdir(gdir) or (classes and genre not in classes):
            continue
        for f in os.listdir(gdir):
            if f.endswith(('.au', '.wav')):
                buckets.setdefault(genre, []).append(os.path.join(gdir, f))
    return _balance(buckets, n_per_class, seed)


# ---- machinery: DCASE2020 Task 2 (pump) ------------------------------------- #
_DCASE_URL = ('https://zenodo.org/records/3678171/files/dev_data_pump.zip')


def _prepare_dcase_pump(root: str) -> str:
    d = os.path.join(root, 'dcase2020_pump')
    inner = os.path.join(d, 'pump')
    if not os.path.isdir(inner):
        arc = download_file(_DCASE_URL, os.path.join(root, 'dcase_pump.zip'))
        print('    extracting…')
        _extract(arc, d)
    return inner


def _collect_dcase_pump(data_dir, classes, n_per_class, seed):
    # The labeled split is test/ with normal_*.wav and anomaly_*.wav.
    test_dir = os.path.join(data_dir, 'test')
    buckets: dict[str, list[str]] = {}
    search = test_dir if os.path.isdir(test_dir) else data_dir
    for r, _, files in os.walk(search):
        for f in files:
            if not f.endswith('.wav'):
                continue
            label = 'anomaly' if f.startswith('anomaly') else (
                'normal' if f.startswith('normal') else None)
            if label is None or (classes and label not in classes):
                continue
            buckets.setdefault(label, []).append(os.path.join(r, f))
    return _balance(buckets, n_per_class, seed)


# --------------------------------------------------------------------------- #
def _balance(buckets: dict[str, list[str]], n_per_class: int | None,
             seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    examples: list[tuple[str, str]] = []
    for label in sorted(buckets):
        paths = sorted(buckets[label])
        rng.shuffle(paths)
        if n_per_class:
            paths = paths[:n_per_class]
        examples.extend((p, label) for p in paths)
    rng.shuffle(examples)
    return examples


DOMAINS: dict[str, DomainSpec] = {
    'speech': DomainSpec(
        name='speech', preset='speech', title='Speech & Paralinguistics',
        task='Speech-emotion classification (RAVDESS)',
        dataset_name='RAVDESS (speech audio, 24 actors)',
        dataset_url=_RAVDESS_URL, prepare=_prepare_ravdess,
        collect=_collect_ravdess,
        default_classes=['neutral', 'happy', 'sad', 'angry'],
        approx_download_mb=210),
    'bioacoustics': DomainSpec(
        name='bioacoustics', preset='bioacoustics', title='Bioacoustics',
        task='Animal-sound classification (ESC-50 animal subset)',
        dataset_name='ESC-50 (environmental sound, animal categories)',
        dataset_url=_ESC50_URL, prepare=_prepare_esc50,
        collect=_collect_esc50, default_classes=_ESC50_ANIMALS,
        approx_download_mb=600),
    'music': DomainSpec(
        name='music', preset='music', title='Music & MIR',
        task='Music-genre classification (GTZAN)',
        dataset_name='GTZAN (10 genres)',
        dataset_url=_GTZAN_URL, prepare=_prepare_gtzan,
        collect=_collect_gtzan,
        default_classes=['classical', 'hiphop', 'jazz', 'metal', 'pop'],
        approx_download_mb=1200),
    'machinery': DomainSpec(
        name='machinery', preset='machinery',
        title='Machinery & Condition Monitoring',
        task='Normal vs. anomaly detection (DCASE2020 Task 2, pump)',
        dataset_name='DCASE2020 Task 2 dev set (pump)',
        dataset_url=_DCASE_URL, prepare=_prepare_dcase_pump,
        collect=_collect_dcase_pump,
        default_classes=['normal', 'anomaly'],
        approx_download_mb=1030),
}


def collect_examples(spec: DomainSpec, classes: Sequence[str] | None, n_per_class: int | None, seed: int,
                     synthetic_dir: str, sr: int) -> tuple[list[tuple[str, str]], str]:
    """
    ((path, label) examples, data source name) for a domain.

    Uses the domain's dataset under ``$DATASETS``. Falls back to 12 synthetic
    2 s clips per class, written to ``synthetic_dir``, when DATASETS is unset,
    AFX_SYNTHETIC=1, or the download or extraction fails.
    """
    root = datasets_root()
    if root and os.environ.get('AFX_SYNTHETIC') != '1':
        try:
            data_dir = spec.prepare(root)
            return spec.collect(data_dir, classes, n_per_class, seed), spec.dataset_name
        except (OSError, zipfile.BadZipFile, tarfile.TarError) as e:
            print(f'    {spec.dataset_name} unavailable ({e}); using synthetic clips')
    paths, labels = write_synthetic_folder(synthetic_dir, n_per_class=min(n_per_class or 12, 12), seconds=2.0,
                                           sr=sr, seed=seed)
    return list(zip(paths, labels.tolist(), strict=True)), 'synthetic stand-in clips (demo_data)'


# --------------------------------------------------------------------------- #
# Feature matrix construction
# --------------------------------------------------------------------------- #
def build_feature_matrix(examples: Sequence[tuple[str, str]],
                         config: dict,
                         max_seconds: float | None = 8.0,
                         verbose: bool = True
                         ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract preset features for every (path, label) example.

    Args:
        examples: list of (audio_path, label).
        config: an AFX config dict (e.g. from ``afx_audio.presets.load_preset``).
        max_seconds: read at most this many seconds of each clip (None = full).
        verbose: print progress.
    Returns:
        (X, y, feature_names) with X of shape (n_examples, n_features). X keeps
        NaN where a feature is undefined over a whole clip; impute it in the
        model pipeline.
    Raises:
        ValueError: a clip gives a different feature layout than the first
            one (a clip shorter than a frame changes some output sizes).
    """
    sr_target = config['sample_rate']
    rows: list[np.ndarray] = []
    labels: list[str] = []
    names: list[str] | None = None
    n = len(examples)
    for i, (path, label) in enumerate(examples):
        try:
            signal, sr = load_audio(path, sr=sr_target, duration=max_seconds)
        except (RuntimeError, ValueError) as e:
            if verbose:
                print(f'    [skip] {os.path.basename(path)}: {e}')
            continue
        clip_names, vec = flatten_features(extract_all_features(signal, sr, config))
        if names is None:
            names = clip_names
        elif clip_names != names:
            raise ValueError(
                f'{path} gives {len(clip_names)} feature columns, the first clip gave {len(names)}. '
                'Drop clips shorter than the longest frame_size in the config or lower max_seconds.'
            )
        rows.append(vec)
        labels.append(label)
        if verbose and (i + 1) % 25 == 0:
            print(f'    extracted {i + 1}/{n}')
    if not rows:
        raise RuntimeError('No features extracted; check the dataset path.')
    return np.vstack(rows), np.asarray(labels), (names or [])


__all__ = [
    'datasets_root', 'download_file', 'DomainSpec', 'DOMAINS',
    'collect_examples', 'build_feature_matrix',
]
