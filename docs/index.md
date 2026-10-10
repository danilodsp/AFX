# AFX

AFX turns audio clips into fixed-length feature vectors for machine
learning. It implements 45 time-domain, autocorrelation, spectral,
spectrogram, cepstral, harmonic and linear-prediction features with NumPy and SciPy, selects
and parameterizes them from a JSON config, and aggregates the per-frame
values into one vector per clip. The same config drives a scikit-learn
transformer, a parallel batch extractor with a CLI, and (for 20 features)
a differentiable PyTorch module.

## Install

```bash
pip install afx-audio            # numpy, scipy, soundfile
pip install "afx-audio[all]"     # + scikit-learn, torch, pandas/pyarrow, matplotlib
```

The import name is `AFX`. Extras: `sklearn`, `torch`, `parquet`, `plot`,
`all`, and `dev` for tests and docs.

## One clip

```python
import AFX

config = AFX.load_preset('speech')
signal, sr = AFX.load_audio('clip.wav', sr=config['sample_rate'])
features = AFX.extract_all_features(signal, sr, config)
names, vector = AFX.flatten_features(features)
```

`features` maps names such as `mfcc_mean` to arrays; `flatten_features`
turns it into column names and one float vector. For a long recording,
`names, X, starts = AFX.extract_segments(signal, sr, config, segment_length=1.0, hop=0.5)`
gives one such vector per segment as the rows of `X`, with each segment's
start time in seconds.

Single features are plain functions:

```python
mfcc = AFX.FEATURE_MAP['mfcc'](signal, sr, n_mfcc=20, frame_size=400, hop_length=160)['mfcc']
AFX.feature_info('mfcc')        # group, units, parameters and defaults, definition
AFX.list_features('spectral')
```

## A folder

```bash
afx dataset/ --preset speech --output features.parquet --jobs 8
```

writes one row per file, keyed by the path relative to `dataset/`. Files
that fail to load are listed in `features.parquet.errors.json`.

## Where to go next

[Configuration](configuration.md) covers framing, aggregation and the
config schema; [Features](features.md) lists every feature with its units
and parameters; [librosa equivalence](librosa.md) lists which features
reproduce librosa exactly and where AFX differs.
