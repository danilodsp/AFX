# AFX

Audio feature extraction for machine learning. 38 time-domain, spectral, cepstral and harmonic features, computed with NumPy and SciPy, driven by a JSON config, with a scikit-learn transformer, a differentiable PyTorch path and a parallel batch CLI on top.

[![Tests](https://github.com/danilodsp/AFX/actions/workflows/test.yml/badge.svg)](https://github.com/danilodsp/AFX/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

```python
import AFX

config = AFX.load_preset('speech')                     # or AFX.load_config('my_config.json')
signal, sr = AFX.load_audio('clip.wav', sr=config['sample_rate'])
features = AFX.extract_all_features(signal, sr, config)  # {'mfcc_mean': (20,), 'mfcc_std': (20,), 'pitch_mean': (1,), ...}
```

## Install

```bash
pip install git+https://github.com/danilodsp/AFX.git
```

Extras: `AFX[torch]` for the PyTorch path, `AFX[parquet]` for Parquet output, `AFX[notebooks]` for the demo notebooks (librosa, seaborn, jupyter), `AFX[dev]` for all of the above plus pytest and ruff. For a local checkout use `pip install -e ".[dev]"`.

## Why AFX

librosa, torchaudio and openSMILE each cover part of this space. AFX is for the case where you want a fixed-length feature vector per clip, selected from a config, with the same numbers on the NumPy, PyTorch and scikit-learn paths and no compiled dependencies beyond NumPy/SciPy. The spectral and cepstral features are tested against librosa (`tests/test_frequency_domain_vs_librosa.py`, `tests/test_cepstral_features.py`), and `notebooks/feature_benchmark_validation.ipynb` reports tolerances against librosa, torchaudio and openSMILE.

## Features

Every feature is a function `extract_<name>(signal, sr, **params)` returning `{name: ndarray}`, registered in `AFX.FEATURE_MAP`. Per-frame shapes are `(n_frames,)` or `(n_dims, n_frames)`; `extract_all_features` aggregates them over time unless `preserve_shape` is set.

| Group | Features | Parameters |
|-------|----------|------------|
| Time domain | `zcr`, `rms_energy`, `short_time_energy`, `energy_ratio`, `variance`, `kurtosis`, `crest_factor`, `entropy`, `sample_entropy`, `mobility`, `complexity` (Hjorth) | `frame_size`, `hop_length` for frame-wise output; omit both for one value per clip |
| Autocorrelation | `autocorr_variance`, `autocorr_snr`, `autocorr_zero_crossings`, `autocorr_peak_width`, `autocorr_decay_rate`, `autocorr_coeffs_deviation`, `max_peak_autocorr_envelope` | `max_lag` (default 100), plus the frame parameters |
| Spectral | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_entropy`, `spectral_flux`, `spectral_contrast`, `spectral_skewness`, `spectral_slope` | `frame_size`, `hop_length`, `center`; `roll_percent`; `n_bands`, `fmin`, `quantile` |
| Cepstral, spectrogram | `mfcc`, `mfcc_delta`, `mfcc_delta_delta`, `gfcc`, `melspectrogram`, `chroma_stft`, `chroma_cqt`, `cqt` | `n_mfcc`, `n_mels`, `fmin`, `fmax`, `center`; `n_gfcc`, `n_gammatone`; `n_bins`, `bins_per_octave` |
| Harmonic | `pitch` (YIN), `hnr`, `thd` | `frame_size`, `hop_length`, `fmin`, `fmax` |

Run `afx --list-features` for the names, or read the docstrings in `AFX/extractors/` for definitions and references.

## Config

A config is a JSON object (or the equivalent dict) with `sample_rate`, a `features` block mapping feature names to extractor parameters, and optional aggregation settings. `AFX/config.json` enables everything; `AFX/presets/*.json` are smaller domain packs.

```json
{
  "sample_rate": 16000,
  "features": {
    "mfcc": {"n_mfcc": 20, "frame_size": 400, "hop_length": 160},
    "pitch": {"frame_size": 400, "hop_length": 160},
    "rms_energy": {}
  },
  "aggregation": "summary",
  "normalize": null
}
```

`aggregation` is one of `mean`, `std`, `summary` (mean and std), `advanced_summary` (mean, std, skew, kurtosis), `skew`, `kurtosis`, `sliding_window` (with `window_size`, `hop_length` and `sliding_agg` under `aggregation_params`) or `stack` (concatenate everything into one vector, no aggregation). `flatten` (default true) flattens each aggregated feature to 1-D. `normalize` is `zscore`, `minmax` or null; without `aggregation_params.norm_stats` it rescales each clip by its own statistics, so for training pipelines leave it null and scale in scikit-learn. `preserve_shape: true` returns the per-frame arrays instead.

`AFX.load_config` and `AFX.validate_config` reject unknown feature names, bad aggregation or normalize values and non-object feature parameters, so typos fail at load time rather than mid-extraction.

### Presets

```python
from AFX.presets import list_presets, load_preset, preset_info
list_presets()   # ['bioacoustics', 'machinery', 'music', 'speech']
```

| Pack | Sample rate | Emphasis |
|------|-------------|----------|
| `speech` | 16 kHz | 20 MFCC with deltas, pitch, HNR, energy, spectral contrast; `summary` aggregation |
| `bioacoustics` | 32 kHz | mel and MFCC, spectral shape, flatness, entropy |
| `music` | 22.05 kHz | chroma (STFT and CQT), spectral contrast, MFCC |
| `machinery` | 16 kHz | frame-wise crest factor, kurtosis, Hjorth parameters, HNR; `advanced_summary` aggregation |

`notebooks/domain_feature_packs_demo.ipynb` runs each pack end to end on an open dataset. Copy a preset JSON to build your own.

### Frame times

With `return_metadata=True`, every extractor returns `metadata['times']`: the time of each frame's center in the original signal, equal to `librosa.frames_to_time(i, sr, hop_length)` for `center=True` and `librosa.frames_to_time(i, sr, hop_length, n_fft=frame_size)` for `center=False`. Features computed with the same `frame_size` and `hop_length` therefore share one time axis regardless of `center`. `spectral_flux[i]` compares frames i and i+1 and carries the time of frame i+1.

`center` defaults to `False` for MFCC, GFCC and the spectral features and to `True` for `melspectrogram`, `chroma_stft`, `chroma_cqt` and `cqt` (matching librosa). Set it explicitly in the config when frame counts must match across features.

## scikit-learn

`AFXFeatureExtractor` accepts file paths or 1-D signals and produces `(n_samples, n_features)` matrices, so it works inside `Pipeline`, `GridSearchCV` and `cross_val_score`. `n_jobs` parallelizes over samples with joblib.

```python
from AFX.sklearn import AFXFeatureExtractor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    AFXFeatureExtractor(features=['mfcc', 'spectral_centroid', 'zcr'],
                        sample_rate=16000, aggregation='summary', n_jobs=-1),
    StandardScaler(),
    SVC(),
)
pipe.fit(train_paths, y_train)
pipe.predict(test_paths)
pipe[0].get_feature_names_out()   # ['zcr_mean', 'zcr_std', 'mfcc_mean_0', ...]
```

## PyTorch

`AFX.torch.functional` reimplements the STFT, mel, MFCC and spectral descriptors in pure torch with the same windowing, padding and filterbank conventions as the NumPy path (parity is tested to 1e-6 or tighter). Inputs are `(time,)` or `(batch, time)` tensors on any device; gradients flow back to the waveform except through the counting and selection steps of `zero_crossing_rate` and `spectral_rolloff`.

```python
import torch
from AFX.torch import AFXFeatureModule, functional as F_afx

mfcc = F_afx.mfcc(waveform, sr=16000, n_mfcc=13)

frontend = AFXFeatureModule(
    features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
    sample_rate=16000, aggregation='mean_std', output='vector',
).to('cuda')
model = torch.nn.Sequential(frontend, torch.nn.Linear(28, 2))
```

Windows, mel filterbanks and DCT matrices are registered as buffers, so they move with `.to(device)`. `AFX.torch.supported_features()` lists the ten features available on this path; the rest stay on the NumPy API. `import AFX` does not import torch.

## Batch extraction and CLI

`extract_features_from_folder` walks a folder recursively, extracts one vector per file with a config path or dict, and writes `npy`, `json`, `csv` or `parquet`. Rows are keyed by path relative to the folder, vector features expand to `name_0, name_1, ...` columns, and files that fail are reported in `<output>.errors.json` instead of aborting the run. `n_jobs` fans out over processes with output identical to the serial run.

```python
from AFX import extract_features_from_folder

extract_features_from_folder('dataset/', 'AFX/presets/speech.json', 'features.parquet', n_jobs=8)
```

The same operation from the shell:

```bash
afx dataset/ --preset speech --output features.parquet --jobs 8
afx dataset/ --config my_config.json --features mfcc,zcr --aggregation summary --output features.csv
afx --list-features
```

The output format follows the `--output` extension unless `--format` is given. `--sample-rate` overrides the config, `--no-flatten` keeps aggregated arrays 2-D, `--normalize zscore|minmax` applies per-file normalization, `-q` silences progress.

## Project layout

```
AFX/
  extractors/     one module per feature group (time_domain, frequency_domain, cepstral_features, harmonic_features)
  methods/        STFT, mel, DCT, delta, CQT, chroma, gammatone, autocorrelation
  torch/          functional transforms and AFXFeatureModule
  utils/          framing, aggregation, normalization, config validation, selectors, plotting
  presets/        domain packs (JSON)
  sklearn.py      AFXFeatureExtractor
  batch_extractor.py, cli.py
notebooks/        demos and the librosa/torchaudio/openSMILE benchmark
tests/            pytest suite, including parity tests against librosa
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check AFX tests
```

## Acknowledgment

The audio examples used in the notebooks are derived from the LibriSpeech corpus ([Panayotov et al., 2015](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)), released under a public domain-compatible license ([download](https://www.openslr.org/12)). No audio files are committed to this repository.

## License

MIT, see [LICENSE](LICENSE).
