# AFX (Audio Feature Extraction)

This project can be used for audio research in fields such as machine learning, sound classification, speech analysis, and music information retrieval.

---

## Features List

- **Standard Audio Features**
  - Mel-Spectrogram
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Delta MFCC
  - Delta-Delta MFCC
  - Zero Crossing Rate (ZCR)
  - Variance
  - Entropy (Sample Entropy)
  - Crest Factor
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Rolloff
  - Chroma Features (STFT-based)
  - Chroma Features (CQT-based)
  - Constant-Q Transform (CQT)
  - Pitch (Fundamental Frequency)
  - Total Harmonic Distortion (THD)
  - Harmonic to Noise Ratio (HNR)

- **Additional Implemented Features**
  - Spectral Contrast
  - Spectral Flatness
  - Spectral Entropy
  - Spectral Flux
  - Kurtosis
  - Short-Time Energy
  - Energy Ratio
  - Autocorrelation-based features (variance, SNR, zero crossings, peak width, decay rate, coeffs deviation, max peak envelope)
  - Mobility
  - Complexity
  - Sample Entropy (alternative implementation)

- **Support for ML/DL Pipelines**
  - Frame-level and file-level feature aggregation, with options for flattening, stacking, and normalization (z-score, min-max)
  - Compatible with numpy arrays and torch tensors
  - Custom windowing and hop-length settings
  - Differentiable, GPU-capable PyTorch path (`AFX.torch`) with an `nn.Module` wrapper for use inside training graphs
  - scikit-learn compatible transformer (`AFX.sklearn.AFXFeatureExtractor`) for Pipelines, GridSearchCV, and cross-validation

- **I/O & Formats**
  - WAV, MP3, FLAC, OGG (via `soundfile`)
  - Batch processing via folder-based loaders, with optional multiprocessing (`n_jobs`)
  - Dataset-scale output to Parquet / CSV / JSON / NPY with named feature columns

---

## Project Structure

```
AFX/
│
├── AFX/                      # Core package
│   ├── extractors/           # Each group features extractor is a modular class
│   ├── io/                   # Audio loading and format handling
│   ├── methods/              # Signal processing algorithms
│   ├── torch/                # Differentiable PyTorch path (optional dep)
│   ├── utils/                # Helpers (e.g. windowing, normalization)
│   ├── sklearn.py            # scikit-learn compatible transformer
│   └── __init__.py
│
├── notebooks/                # Jupyter notebooks for examples
├── examples/                 # Script examples
├── tests/                    # Unit tests
├── requirements.txt
├── README.md
└── pyproject.toml
```

---

## Getting Started

```bash
git clone https://github.com/danilodsp/AFX.git
cd AFX

python -m venv venv
source venv/bin/activate    # macOS/Linux
venv\Scripts\activate       # Windows

pip install -e .
```

To also install the dependencies used by the example notebooks (seaborn, librosa, jupyter):

```bash
pip install -e ".[notebooks]"
```

---


## Example Usage

```python
from AFX import extract_all_features
from AFX.utils.config_loader import load_config
from AFX.utils.aggregator import aggregate_features
from AFX.io.io import load_audio

config = load_config('AFX/config.json')
signal, sr = load_audio('path/to/audio.wav', sr=config['sample_rate'])

# 1. extract individual features (original, per-frame shape)
features = extract_all_features(signal, sr=sr, config=config)
# features is a dict of per-frame feature arrays

# 2. aggregate, flatten, and z-score normalize features
agg_features = aggregate_features(features, method='mean', flatten=True, normalize='zscore')
# agg_features is a dict of 1-D arrays (all features in dict flattened, concatenated and normalized)

# 3. aggregate features (stacked)
agg_stacked = aggregate_features(features, method='stack')
# agg_stacked is a 1D numpy array (all features stacked, no normalization)
```

Another way of implementing them:

```python
# per-frame features (original shape)
config['preserve_shape'] = True
features = extract_all_features(signal, sr, config)

# aggregated, flattened, and normalized features
config['preserve_shape'] = False
config['aggregation'] = 'mean'
config['flatten'] = True
config['normalize'] = 'zscore'
agg_features = extract_all_features(signal, sr, config)

# aggregated, stacked (not flattened) features
config['preserve_shape'] = False
config['aggregation'] = 'stack'
agg_stacked = extract_all_features(signal, sr, config)
```

Check the [Notebooks](notebooks/) folder for more examples.

## Domain feature packs

AFX ships small, curated **presets** tuned for common audio domains, so you can
skip hand-picking features and start from a sensible, domain-appropriate set.
Each pack is a validated config (same schema as `AFX/config.json`) that drops
into any AFX entry point.

```python
from AFX.presets import list_presets, load_preset, preset_info

list_presets()
# ['bioacoustics', 'machinery', 'music', 'speech']

config = load_preset('speech')      # a validated AFX config dict
features = extract_all_features(signal, sr=config['sample_rate'], config=config)
```

| Pack           | Focus                                   | Highlights                                             |
|----------------|-----------------------------------------|--------------------------------------------------------|
| `speech`       | Speech & paralinguistics (emotion, voice quality) | 16 kHz, MFCC + Δ + ΔΔ, pitch, HNR, energy         |
| `bioacoustics` | Animal / environmental sound            | 32 kHz, mel + MFCC, spectral shape, flatness, entropy  |
| `music`        | Music information retrieval             | chroma (CQT + STFT), spectral contrast, MFCC           |
| `machinery`    | Condition monitoring / fault detection  | framewise crest factor, kurtosis, Hjorth params, HNR   |

The [`domain_feature_packs_demo.ipynb`](notebooks/) notebook runs each pack
end-to-end on a small open dataset (download → extract → classify → plot) — set
`DOMAIN` and Run All. Build your own pack by copying a JSON file in
`AFX/presets/` and picking the features that matter for your problem.

## scikit-learn integration

`AFXFeatureExtractor` is a standard sklearn transformer: it accepts file
paths or raw 1-D signals and produces fixed-length feature matrices, so it
drops straight into Pipelines and cross-validation:

```python
from AFX.sklearn import AFXFeatureExtractor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    AFXFeatureExtractor(features=['mfcc', 'spectral_centroid', 'zcr'],
                        sample_rate=16000, n_jobs=-1),
    StandardScaler(),
    SVC(),
)
pipe.fit(train_wav_paths, y_train)   # X can be paths or raw signals
pipe.predict(test_wav_paths)
```

`get_feature_names_out()` returns expanded column names (`mfcc_0`,
`mfcc_1`, ..., `spectral_centroid`), and `n_jobs` parallelizes extraction
over samples via joblib. For train/test-consistent scaling, prefer
`normalize=None` (default) plus a `StandardScaler` pipeline step.

## PyTorch path (differentiable / GPU)

`AFX.torch` provides pure-torch implementations of the core transforms
(STFT, mel, MFCC, spectral descriptors) that numerically match the NumPy
path, run on CPU or CUDA, and are differentiable end-to-end — install with
`pip install "AFX[torch]"`:

```python
import torch
from AFX.torch import AFXFeatureModule, functional as F_afx

# Functional API: works on (time,) or batched (batch, time) tensors
mfcc = F_afx.mfcc(waveform, sr=16000, n_mfcc=13)          # differentiable
mel = F_afx.melspectrogram(waveform.cuda(), sr=16000)     # GPU

# nn.Module wrapper: usable as a trainable front end
frontend = AFXFeatureModule(
    features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
    sample_rate=16000, aggregation='mean', output='vector',
).to('cuda')
model = torch.nn.Sequential(frontend, torch.nn.Linear(14, 2))
```

Constants (windows, mel filterbanks, DCT matrices) are registered as
buffers, so they follow `.to(device)` and add no per-forward overhead.
`AFX.torch.supported_features()` lists what's available on this path;
everything else remains available via the NumPy API. PyTorch stays an
optional dependency — `import AFX` works without it.

## Parallel batch extraction

`extract_features_from_folder` (and the CLI) can fan extraction out over
worker processes and write dataset-friendly formats:

```python
from AFX import extract_features_from_folder

extract_features_from_folder(
    'path/to/dataset', 'AFX/config.json', 'features.parquet',
    save_format='parquet',   # or 'csv', 'json', 'npy'
    n_jobs=8,                # 0 = all cores; results identical to serial
)
```

Parquet output (one row per file, columns like `mfcc_0 ... mfcc_12`,
`zcr`, ...) requires `pip install "AFX[parquet]"` and loads directly into
pandas: `pd.read_parquet('features.parquet')`.

## CLI

Installing the package provides an `afx` command for batch feature extraction over a folder of audio files:

```bash
afx <input_folder> [--config path/to/config.json] [--output features.npy] \
    [--format npy|json|csv|parquet] [--features mfcc,zcr,spectral_centroid] \
    [--aggregation mean] [--normalize zscore] [--no-flatten] [--jobs 8]
```

- `input_folder`: folder containing audio files to process
- `--config`: path to a config JSON (defaults to the bundled `AFX/config.json`)
- `--output`: output file path (default `features.npy`)
- `--format`: output format, one of `npy`, `json`, `csv` (default `npy`)
- `--features`: comma-separated list of features to extract (default: all in config)
- `--aggregation`: aggregation method (e.g. `mean`)
- `--normalize`: normalization method, `zscore` or `minmax`
- `--no-flatten`: disable flattening after aggregation
- `--jobs` / `-j`: number of parallel worker processes (`0` = all cores, default `1`)

## Acknowledgment

- The audio examples used in the notebooks are derived from the LibriSpeech corpus ([Panayotov et al., 2015](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)), which is released under a public domain-compatible license ([download link](https://www.openslr.org/12)). No audio files are committed to this repository.
