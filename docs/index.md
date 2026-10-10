# AFX

AFX turns audio clips into fixed-length feature vectors for machine
learning. It implements 47 time-domain, autocorrelation, spectral,
spectrogram, cepstral, harmonic and linear-prediction features and the
eGeMAPS v02 parameter set with NumPy and SciPy, selects
and parameterizes them from a JSON config, and aggregates the per-frame
values into one vector per clip. The same config drives a scikit-learn
transformer, a parallel batch extractor with a CLI, and (for 20 features)
a differentiable PyTorch module.

## Install

```bash
pip install afx-audio            # numpy, scipy, soundfile
pip install "afx-audio[all]"     # + scikit-learn, torch, pandas/pyarrow, matplotlib
```

The import name is `afx_audio` (`import afx_audio as afx`). Extras: `sklearn`, `torch`, `parquet`, `plot`,
`all`, and `dev` for tests and docs.

## One clip

```python
import afx_audio as afx

config = afx.load_preset('speech@1')
signal, sr = afx.load_audio('clip.wav', sr=config['sample_rate'])
features = afx.extract_all_features(signal, sr, config)
names, vector = afx.flatten_features(features)
```

`features` maps names such as `mfcc_mean` to arrays; `flatten_features`
turns it into column names and one float vector. For a long recording,
`names, X, starts = afx.extract_segments(signal, sr, config, segment_length=1.0, hop=0.5)`
gives one such vector per segment as the rows of `X`, with each segment's
start time in seconds.

Single features are plain functions:

```python
mfcc = afx.FEATURE_MAP['mfcc'](signal, sr, n_mfcc=20, frame_size=400, hop_length=160)['mfcc']
afx.feature_info('mfcc')        # group, units, parameters and defaults, definition
afx.list_features('spectral')
```

## A folder

```bash
afx dataset/ --preset speech --output features.parquet --jobs 8
```

writes one row per file, keyed by the path relative to `dataset/`. Files
that fail to load are listed in `features.parquet.errors.json`, and the
Parquet metadata records the config hash, preset id, library versions and
input file hashes ([Reproducibility](reproducibility.md)).

## Where to go next

[Configuration](configuration.md) covers framing, aggregation, coverage,
presets and the config schema; [Features](features.md) lists every feature
with its units, parameters and undefined cases; [eGeMAPS](egemaps.md)
compares the parameter set with openSMILE; [librosa equivalence](librosa.md)
lists which features reproduce librosa exactly and where AFX differs;
[Notebooks](notebooks.md) is the tutorial.
