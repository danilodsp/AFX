# AFX

Audio feature extraction for machine learning. 45 time-domain, autocorrelation, spectral, cepstral, harmonic and linear-prediction features computed with NumPy and SciPy, selected from a JSON config and aggregated into one vector per clip, with a scikit-learn transformer, a differentiable PyTorch path and a parallel batch CLI on top.

[![Tests](https://github.com/danilodsp/AFX/actions/workflows/test.yml/badge.svg)](https://github.com/danilodsp/AFX/actions/workflows/test.yml)
![Python](https://img.shields.io/badge/python-3.10%20to%203.14-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Documentation: [danilodsp.github.io/AFX](https://danilodsp.github.io/AFX/), built from [docs/](https://github.com/danilodsp/AFX/tree/main/docs/).

```python
import afx_audio as afx

config = afx.load_preset('speech')                       # or afx.load_config('my_config.json')
signal, sr = afx.load_audio('clip.wav', sr=config['sample_rate'])
features = afx.extract_all_features(signal, sr, config)  # {'mfcc_mean': (20,), 'mfcc_std': (20,), ...}
names, vector = afx.flatten_features(features)           # column names and one float vector
```

## Install

```bash
pip install afx-audio
```

The distribution is `afx-audio` and the import name is `afx_audio`; the examples use `import afx_audio as afx`. The core install needs only NumPy, SciPy and soundfile. Extras: `sklearn` for `AFXFeatureExtractor` and the selectors, `torch` for the PyTorch path, `parquet` (pandas and pyarrow) for Parquet output, `plot` for `afx.utils.visualization`, `all` for those four, `docs`, `notebooks`, and `dev` for everything plus pytest and ruff. From a checkout: `pip install -e ".[dev]"`.

## Why AFX

librosa, torchaudio and openSMILE each cover part of this space. AFX is for the case where you want a fixed-length feature vector per clip, chosen from a config that fails fast on typos, with the same numbers on the NumPy, PyTorch and scikit-learn paths and no compiled dependencies beyond NumPy and SciPy. Fourteen features reproduce a librosa function to float32 precision or better (`tests/test_librosa_parity.py`, listed in [docs/librosa.md](https://github.com/danilodsp/AFX/blob/main/docs/librosa.md)).

## Features

Every feature is a function `extract_<name>(signal, sr, **params)` returning `{name: ndarray}`, registered in `afx.FEATURE_MAP`. Per-frame shapes are `(n_frames,)` or `(n_dims, n_frames)`. `afx.feature_info(name)` gives a feature's definition, units and parameters with defaults; [docs/features.md](https://github.com/danilodsp/AFX/blob/main/docs/features.md) lists them all. `afx.register_feature(name, func)` adds your own extractor to configs and every front end.

| Group | Features |
|-------|----------|
| Time domain | `zcr`, `rms_energy`, `short_time_energy`, `energy_ratio`, `variance`, `kurtosis`, `crest_factor`, `entropy`, `sample_entropy`, `mobility`, `complexity` |
| Autocorrelation | `autocorr_variance`, `autocorr_snr`, `autocorr_zero_crossings`, `autocorr_peak_width`, `autocorr_decay_rate`, `autocorr_coeffs_deviation`, `max_peak_autocorr_envelope` |
| Spectral | `spectral_centroid`, `spectral_bandwidth`, `spectral_rolloff`, `spectral_flatness`, `spectral_entropy`, `spectral_flux`, `spectral_contrast`, `spectral_skewness`, `spectral_kurtosis`, `spectral_slope`, `spectral_crest`, `spectral_decrease` |
| Spectrogram | `melspectrogram`, `cqt`, `chroma_stft`, `chroma_cqt`, `tonnetz`, `onset_strength` |
| Cepstral | `mfcc`, `mfcc_delta`, `mfcc_delta_delta`, `gfcc` |
| Harmonic | `pitch` (YIN), `hnr`, `thd` |
| Linear prediction | `lpc`, `formants` |

Time-domain and autocorrelation features take `frame_size`, `hop_length` and `center` for frame-wise output, or analyze the whole clip when the sizes are omitted. Spectral and cepstral features use an STFT with a periodic Hann window, `n_fft = frame_size`, and optional librosa-style `center` padding. With `return_metadata=True` every extractor also returns the time of each frame's center, so features with the same framing share one time axis.

## Config

```json
{
  "sample_rate": 16000,
  "features": {
    "mfcc": {"n_mfcc": 20, "frame_size": 400, "hop_length": 160},
    "pitch": {"frame_size": 400, "hop_length": 160},
    "rms_energy": {}
  },
  "aggregation": "summary"
}
```

`aggregation` is `mean`, `std`, `summary` (mean and std), `advanced_summary` (plus skewness and kurtosis), `skew`, `kurtosis`, `sliding_window` or `stack`. Frames where a feature is undefined (HNR on silence) are NaN and skipped by the statistics. `afx.validate_config` rejects unknown features, unknown parameters and non-positive sizes before any audio is read. Bundled presets: `speech`, `bioacoustics`, `music`, `machinery` (`afx.list_presets()`). See [docs/configuration.md](https://github.com/danilodsp/AFX/blob/main/docs/configuration.md) for framing, aggregation and normalization details.

## scikit-learn

```python
from afx_audio.sklearn import AFXFeatureExtractor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = make_pipeline(
    AFXFeatureExtractor(features=['mfcc', 'spectral_centroid', 'zcr'],
                        sample_rate=16000, aggregation='summary', n_jobs=-1),
    StandardScaler(),
    SVC(),
)
pipe.fit(train_paths, y_train)       # file paths or 1-D signals
pipe[0].get_feature_names_out()      # ['mfcc_mean_0', ..., 'zcr_std']
```

Features that can be undefined for a whole clip (HNR of silence, formants of noise) come out as NaN; put a `SimpleImputer` before the scaler when using them. `examples/classify_folders.py` runs a preset through such a pipeline with cross-validation on a folder per class.

## PyTorch

```python
from afx_audio.torch import AFXFeatureModule, functional as F_afx

mfcc = F_afx.mfcc(waveform, sr=16000, n_mfcc=13)   # (time,) or (batch, time), any device
frontend = AFXFeatureModule(features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
                            sample_rate=16000, aggregation='mean_std', output='vector')
```

Twenty features (mel spectrogram, MFCC and deltas, GFCC, chroma_stft, the twelve spectral descriptors, zcr and RMS) are implemented in torch with the same conventions as the NumPy path, tested against it to 1e-6 in float64, batched, and differentiable with respect to the waveform except through the counting in `zcr` and the bin selection in `spectral_rolloff`. `afx.torch.supported_features()` lists them. `import afx_audio as afx` does not import torch.

## Batch extraction and CLI

```bash
afx dataset/ --preset speech --output features.parquet --jobs 8
afx dataset/ --config my_config.json --features mfcc,zcr --aggregation summary --output features.csv
afx --list-features
afx --describe spectral_rolloff
```

or `afx.extract_features_from_folder(folder, config, output, n_jobs=8)`. The output format follows the extension (`npy`, `json`, `csv`, `parquet`), rows are keyed by the path relative to the folder, in CSV and Parquet vector features expand to `name_0, name_1, ...` columns (JSON and npy keep one list per feature), and files that fail are listed in `<output>.errors.json` instead of aborting the run. Parallel output is identical to serial output. Worker processes use the platform's default start method, so on Windows, macOS and Python 3.14+ a script that calls it with `n_jobs > 1` needs an `if __name__ == '__main__':` guard.

## Performance

`python benchmarks/bench_features.py` times every feature in a config. With the bundled `config.json`, `extract_all_features` on 10 s of audio at 22.05 kHz takes about 0.5 s on one CPU core for everything except `sample_entropy`, which alone takes another 0.4 to 0.8 s (template matching is quadratic in the frame length). Extractors with the same STFT parameters share one memoized STFT.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check AFX tests
mypy AFX
mkdocs serve
```

`python docs/gen_features.py` regenerates the feature reference after an extractor changes; a test fails while it is stale.

## Acknowledgment

The audio examples used in the notebooks are derived from the LibriSpeech corpus ([Panayotov et al., 2015](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)), released under a public domain-compatible license ([download](https://www.openslr.org/12)). No audio files are committed to this repository.

## License

MIT, see [LICENSE](https://github.com/danilodsp/AFX/blob/main/LICENSE).
