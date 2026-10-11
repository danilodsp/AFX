# AFX

Audio feature extraction for machine learning. AFX computes 47 features (time-domain, autocorrelation, spectral, spectrogram, cepstral, harmonic, linear prediction, and the eGeMAPS v02 parameter set) with NumPy and SciPy, selects them from a validated JSON config, and aggregates the frames into one vector per clip. The same config drives a scikit-learn transformer, a parallel batch CLI whose outputs record how they were made, and (for the 20 features with torch implementations) a differentiable PyTorch module.

[![Tests](https://github.com/danilodsp/AFX/actions/workflows/test.yml/badge.svg)](https://github.com/danilodsp/AFX/actions/workflows/test.yml)
[![codecov](https://codecov.io/gh/danilodsp/AFX/branch/main/graph/badge.svg)](https://codecov.io/gh/danilodsp/AFX)
[![PyPI](https://img.shields.io/pypi/v/afx-audio?label=PyPI)](https://pypi.org/project/afx-audio/)
![Python](https://img.shields.io/badge/python-3.10%20to%203.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/danilodsp/AFX/blob/main/LICENSE)

Documentation: [danilopena.com/AFX](https://danilopena.com/AFX/)

```python
import afx_audio as afx

config = afx.load_preset('speech@1')                     # or afx.load_config('my_config.json')
signal, sr = afx.load_audio('clip.wav', sr=config['sample_rate'])
features = afx.extract_all_features(signal, sr, config)  # {'mfcc_mean': (20,), 'mfcc_std': (20,), ...}
names, vector = afx.flatten_features(features)           # column names and one float vector

afx.egemaps(signal, sr)                                  # the 88 eGeMAPS v02 functionals, by name
```

## Install

```bash
pip install afx-audio
pip install "afx-audio[all]"     # + scikit-learn, torch, pandas/pyarrow, matplotlib
```

The import name is `afx_audio`; the examples use `import afx_audio as afx`. The core install needs NumPy, SciPy and soundfile only. Extras: `sklearn` for `AFXFeatureExtractor` and the selectors, `torch` for the PyTorch path, `parquet` (pandas and pyarrow) for Parquet output, `plot` for `afx_audio.utils.visualization`, `all` for those four, `notebooks`, `docs`, and `dev` for everything plus the test and lint tools.

## Why AFX

librosa computes frame-level features; openSMILE computes standard parameter sets under a research-only licence; TSFEL covers generic time series. AFX sits between them for audio ML work:

| Need | AFX |
|---|---|
| One vector per clip from a config | JSON config, validated before any audio is read, with "did you mean" suggestions for typos |
| Standard voice features without licence restrictions | eGeMAPS v02 under MIT, agreement with openSMILE published per functional ([docs/egemaps.md](https://danilopena.com/AFX/egemaps/)) |
| Numbers you can check | 14 features equal librosa to float32 precision or better, 20 torch features equal the NumPy path within rtol 1e-9 in float64, all in the test suite |
| Results you can reproduce and cite | versioned presets (`speech@1`) frozen by hash, and every batch output records config hash, library versions and input file hashes |
| Knowing what a statistic is based on | `"coverage": true` adds the fraction of valid frames next to every feature; `feature_info(name)['undefined']` says when a frame is NaN |
| Pipelines | scikit-learn transformer on file paths or arrays, differentiable torch module, batch CLI to npy/json/csv/parquet |

The core has no compiled dependencies beyond NumPy, SciPy and soundfile.

## Features

Every feature is a function `extract_<name>(signal, sr, *, **params)` returning `{name: ndarray}`, registered in `afx.FEATURE_MAP`. Per-frame shapes are `(n_frames,)` or `(n_dims, n_frames)`, always float64. `afx.feature_info(name)` gives a feature's definition, units, parameters with defaults, when it is undefined, and its torch counterpart; [the feature reference](https://danilopena.com/AFX/features/) lists them all. `afx.register_feature(name, func)` adds your own extractor to configs and every front end.

| Group | Features |
|-------|----------|
| Time domain | `zcr`, `rms_energy`, `short_time_energy`, `energy_ratio`, `variance`, `kurtosis`, `crest_factor`, `entropy`, `sample_entropy`, `mobility`, `complexity` |
| Autocorrelation | `autocorr_variance`, `autocorr_snr`, `autocorr_zero_crossings`, `autocorr_peak_width`, `autocorr_decay_rate`, `autocorr_coeffs_deviation`, `max_peak_autocorr_envelope` |
| Spectral | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_entropy`, `spectral_flux`, `spectral_contrast`, `spectral_skewness`, `spectral_kurtosis`, `spectral_slope`, `spectral_crest`, `spectral_decrease` |
| Spectrogram | `melspectrogram`, `cqt`, `chroma_stft`, `chroma_cqt`, `tonnetz`, `onset_strength` |
| Cepstral | `mfcc`, `mfcc_delta`, `mfcc_delta_delta`, `gfcc` |
| Harmonic | `pitch` (YIN), `hnr`, `thd` |
| Linear prediction | `lpc`, `formants` |
| Parameter set | `egemaps` (88 functionals per clip), `egemaps_lld` (25 descriptors per 10 ms frame) |

Framed features take `frame_size`, `hop_length` and `center` (default True, as in librosa; `onset_strength` is always centered and has no `center`), so features with the same framing share frame count and frame times; time-domain features analyze the whole clip when the sizes are omitted. Spectral features use an STFT with a periodic Hann window and `n_fft = frame_size`. With `return_metadata=True` every extractor also returns the time of each frame's center.

## Config

```json
{
  "sample_rate": 16000,
  "features": {
    "mfcc": {"n_mfcc": 20, "frame_size": 400, "hop_length": 160},
    "pitch": {"frame_size": 400, "hop_length": 160, "fmax": 600},
    "rms_energy": {"frame_size": 400, "hop_length": 160}
  },
  "aggregation": "summary",
  "coverage": true
}
```

`aggregation` is `mean`, `std`, `summary` (mean and std), `advanced_summary` (plus skewness and kurtosis), `skew`, `kurtosis`, `sliding_window` or `stack`. Frames where a feature is undefined (pitch on unvoiced frames, spectral entropy on silence) are NaN and skipped by the statistics, and `coverage` adds `<feature>_valid_frac`. `afx.validate_config` rejects unknown features, unknown parameters and non-positive sizes before any audio is read.

Bundled presets: `speech`, `bioacoustics`, `music`, `machinery` and `egemaps` (`afx.list_presets(versions=True)`). A preset id such as `speech@1` names a frozen file that spells out every parameter; cite it with the package version and the config hash from the output's provenance record. [ESC-50 baselines](https://danilopena.com/AFX/baselines/) give reference accuracies for each preset. See [Configuration](https://danilopena.com/AFX/configuration/) for framing, aggregation, normalization and presets.

## scikit-learn

```python
from afx_audio.sklearn import AFXFeatureExtractor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    AFXFeatureExtractor(config='speech@1', n_jobs=-1),
    SimpleImputer(),
    StandardScaler(),
    SVC(),
)
pipe.fit(train_paths, y_train)       # file paths or 1-D signals
pipe[0].get_feature_names_out()      # ['mfcc_mean_0', ..., 'spectral_contrast_std_6']
pipe[0].provenance_                  # config hash, versions, preset id
```

Features that can be undefined for a whole clip (pitch of noise, formants of silence) come out as NaN, hence the imputer. `examples/classify_folders.py` runs a preset through such a pipeline with cross-validation on a folder per class.

## PyTorch

```python
from afx_audio.torch import AFXFeatureModule, functional as F_afx

mfcc = F_afx.mfcc(waveform, sr=16000, n_mfcc=13)   # (time,) or (batch, time), any device
frontend = AFXFeatureModule(features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
                            sample_rate=16000, aggregation='mean_std', output='vector')
frontend = AFXFeatureModule.from_config('bioacoustics@1', output='vector')  # preset, JSON path or dict
```

Twenty features (mel spectrogram, MFCC and deltas, GFCC, chroma_stft, the twelve spectral descriptors, zcr and RMS) have torch implementations with the NumPy conventions, batched and differentiable with respect to the waveform except through the counting in `zcr` and the bin selection in `spectral_rolloff`. `afx.feature_info(name)['torch']` gives the tolerance each one is tested to. `AFXFeatureModule.from_config` reads the features, sample rate, aggregation and coverage of a config; a config with other features raises unless `skip_unsupported=True`, which drops them with an `AFXWarning`. `import afx_audio` does not import torch.

## Batch extraction and CLI

```bash
afx dataset/ --preset speech@1 --output features.parquet --jobs 8 --stats
afx dataset/ --config my_config.json --features mfcc,zcr --aggregation summary --output features.csv
afx --provenance features.parquet
afx --list-features
afx --describe spectral_rolloff
```

or `afx.extract_features_from_folder(folder, config, output, n_jobs=8)`. The output format follows the extension (`npy`, `json`, `csv`, `parquet`), and `afx` without `--output` writes `features.csv`; every file whose extension is a format soundfile reads (wav, flac, ogg, opus, mp3, aiff, ...; `--ext wav,flac` or `file_exts=` narrows it) is read, and rows are keyed by the path relative to the folder; in CSV and Parquet vector features expand to `name_0, name_1, ...` columns (eGeMAPS columns keep their openSMILE names); files that fail are listed in `<output>.errors.json`. Parallel output equals serial output.

Every output records its provenance (config and config hash, preset id, AFX, NumPy, SciPy and Python versions, sample rate and resampling, sha256 of each input file): in the Parquet schema metadata, otherwise in `<output>.provenance.json`. `--stats` (`dataset_stats=True`) also writes per-column mean, variance, min, max and NaN counts in one streaming pass, and `afx_audio.sklearn.scaler_from_stats` turns them into a fitted `StandardScaler`. See [Reproducibility](https://danilopena.com/AFX/reproducibility/).

Workers never fork (forkserver on Linux, spawn on Windows and macOS), so a script that calls `extract_features_from_folder` with `n_jobs > 1` needs an `if __name__ == '__main__':` guard, and custom features must be importable from a module.

## Notebooks

[notebooks/](https://github.com/danilodsp/AFX/tree/main/notebooks) has a numbered tutorial from `01_quickstart` to `08_egemaps`, each runnable on synthetic audio without a dataset, and `benchmark_validation`, which compares AFX with librosa, torchaudio, openSMILE, spafe and Praat.

## Performance

On a 2-core 2.1 GHz Xeon VM, `extract_all_features` with the bundled `config.json` (39 features) takes 0.57 s for 10 s of audio at 22.05 kHz, of which `sample_entropy` is 0.27 s. Spectral features run at librosa's speed (8 to 12 ms each), and the torch module computes its 20 features for a batch of 16 clips at 0.14 s per clip. `python benchmarks/bench_features.py --librosa --torch-batch 16 --markdown` reproduces the [benchmarks page](https://danilopena.com/AFX/benchmarks/) on your machine.

## Citing

If you use AFX in research, cite it with the metadata in [CITATION.cff](https://github.com/danilodsp/AFX/blob/main/CITATION.cff) (GitHub's "Cite this repository" button), and give the preset id and config hash of the features you used. For eGeMAPS also cite Eyben et al. (2016).

## Development

```bash
pip install -e ".[dev]"
pre-commit install
pytest                  # pytest -m "not slow" for a quick run
ruff check .
mypy afx_audio
mkdocs serve
```

`python docs/gen_features.py` regenerates the feature reference after an extractor changes; a test fails while it is stale. See [CONTRIBUTING.md](https://github.com/danilodsp/AFX/blob/main/CONTRIBUTING.md).

## License

MIT, see [LICENSE](https://github.com/danilodsp/AFX/blob/main/LICENSE). No audio is committed to this repository; the notebooks synthesize clips or read datasets you download yourself.
