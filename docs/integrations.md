# scikit-learn, PyTorch and batch extraction

## scikit-learn

`afx.sklearn.AFXFeatureExtractor` (extra `sklearn`) takes file paths or
1-D signals and returns an `(n_samples, n_features)` array, so it fits in
`Pipeline`, `GridSearchCV` and `cross_val_score`. `n_jobs` parallelizes
over samples with joblib.

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
pipe.fit(train_paths, y_train)
pipe[0].get_feature_names_out()   # ['mfcc_mean_0', ..., 'zcr_std']
```

## PyTorch

`afx.torch` (extra `torch`) reimplements 20 features with torch
operations and the same conventions as the NumPy path: mel spectrogram,
MFCC, MFCC deltas, GFCC, chroma_stft, the twelve spectral descriptors
(centroid, bandwidth, rolloff, entropy, flatness, flux, skewness,
kurtosis, slope, crest, decrease, contrast), zero-crossing rate and RMS. `afx.torch.supported_features()`
lists them, and the tests hold every one to rtol 1e-9 of the NumPy
extractor in float64 at two framings (`afx.feature_info(name)['torch']`
gives each tolerance). Inputs are `(time,)` or
`(batch, time)` tensors on any device. Gradients reach the waveform except
through the counting in `zcr` and the bin selection in `spectral_rolloff`.

```python
import torch
from afx_audio.torch import AFXFeatureModule, functional as F_afx

mfcc = F_afx.mfcc(waveform, sr=16000, n_mfcc=13)

frontend = AFXFeatureModule(
    features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
    sample_rate=16000, aggregation='mean_std', output='vector',
)
model = torch.nn.Sequential(frontend, torch.nn.Linear(28, 2))
```

With `output='vector'`, `frontend.get_feature_names_out()` names the
columns as the NumPy path does, and the vector equals
`flatten_features(extract_all_features(...))` with `aggregation='summary'`
for `'mean_std'`.

`AFXFeatureModule.from_config(config)` builds the module from a config
dict, a JSON path or a preset id, with the config's per-feature
parameters, `sample_rate`, `aggregation` and `coverage`
(`preserve_shape` becomes `aggregation='none'`). Aggregations other than
`mean`, `std` and `summary`, and `normalize`, raise ValueError. So do
features without a torch implementation, unless `skip_unsupported=True`
drops them with an `AFXWarning`. Of the presets only `bioacoustics` is
fully on the torch path; `speech` loses pitch, HNR and formants.

```python
frontend = AFXFeatureModule.from_config('speech@1', output='vector', skip_unsupported=True)
```

Windows, filterbanks and DCT matrices are buffers, so they follow
`.to(device)`. They are stored in float64 and cast to the input dtype at
each forward: float64 input gives float64 output at the precision of the
functions, and float32 input gives float32 output. `.float()` or `.half()`
on the module lowers the stored constants as well. Parameters the torch
path does not implement raise in the constructor.

## Folders and the CLI

`afx.extract_features_from_folder(folder, config, output)` walks a folder
recursively and writes one row per file as `npy` (a pickled dict), `json`,
`csv` or `parquet` (extra `parquet`). Vector features expand to `name_0,
name_1, ...` columns. `n_jobs` runs files in worker processes with output
identical to a serial run. The workers never fork (fork deadlocks when the
parent has threads): they start with forkserver on Linux and spawn on
Windows and macOS, or with the method set by
`multiprocessing.set_start_method`. Each worker imports the calling script,
so a script calling it with `n_jobs > 1` must do so under
`if __name__ == '__main__':`, and registered custom features must be
importable (not defined in a notebook).

```bash
afx dataset/ --preset speech --output features.parquet --jobs 8
afx dataset/ --config my.json --features mfcc,zcr --aggregation summary --output features.csv
afx --list-features
afx --describe spectral_rolloff
```

## Custom features

`afx.register_feature(name, func, units=..., rows=...)` adds a function
`func(signal, sr, **params) -> {name: array}` to `FEATURE_MAP`, so configs,
`validate_config`, `feature_info`, the scikit-learn transformer and the
batch extractor accept it. Wrap a per-frame function with
`afx.utils.framewise.framewise_extractor` to get `frame_size`,
`hop_length`, `center` and frame times for free (it calls Python once per
frame). The batch extractor and `AFXFeatureExtractor` repeat the
registrations in their worker processes, which requires the function to be
picklable: defined at module level, not a lambda or a nested function.

```python
import numpy as np
import afx_audio as afx
from afx_audio.utils.framewise import framewise_extractor

@framewise_extractor
def peak_amplitude(signal, sr=None, *, scale=1.0):
    return {'peak_amplitude': np.array([scale * np.max(np.abs(signal))])}

afx.register_feature('peak_amplitude', peak_amplitude, units='amplitude')
afx.extract_all_features(x, sr, {'features': {'peak_amplitude': {'frame_size': 1024, 'hop_length': 512}}})
```
