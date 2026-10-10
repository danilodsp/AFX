# Configuration

A config is a dict (or a JSON file loaded with `afx.load_config`):

```json
{
  "sample_rate": 16000,
  "features": {
    "mfcc": {"n_mfcc": 20, "frame_size": 400, "hop_length": 160},
    "pitch": {"frame_size": 400, "hop_length": 160},
    "rms_energy": {}
  },
  "aggregation": "summary",
  "flatten": true,
  "normalize": null
}
```

`features` maps feature names to the keyword arguments of their
extractors; a list of names (`"features": ["mfcc", "zcr"]`) uses the
defaults. `afx.validate_config` (called by `load_config`,
`extract_all_features`, the batch extractor and the scikit-learn
transformer) rejects unknown keys, feature names and parameters (with
the closest valid name in the message), non-positive frame sizes and
counts, unknown `aggregation` or `normalize` values, and `sliding_window`
without its window sizes, so a typo fails before any audio is read.
`afx.feature_info(name)['parameters']` lists what each feature accepts.
`sample_rate` is required in config files; `extract_all_features` takes
the rate as an argument and only warns when it differs from the config's.

## Framing

Time-domain, autocorrelation and THD features use frames of `frame_size`
samples every `hop_length` samples; samples after the last full frame are
dropped. With `center=True` the signal is first zero-padded by
`frame_size // 2` on each side (`zcr` repeats the edge samples, as librosa
does), which gives the same frames and times as a centered STFT with the
same sizes. Leave both sizes out (the default for most of them) to get one
value for the whole clip; an uncentered clip shorter than one frame is also
analyzed as a single unpadded frame. `zcr` defaults to 2048/512.

Spectral, spectrogram and cepstral features use the STFT with a periodic
Hann window and `n_fft = frame_size`. With `center=True` the signal is
first zero-padded by `frame_size // 2` on each side, as librosa does
(`spectral_flux` pads by reflection). `center` defaults to `False` for
MFCC, GFCC, deltas and the spectral shape features, and to `True` for
`melspectrogram`, `chroma_stft`, `cqt`, `spectral_contrast` and
`spectral_flux`. Set it explicitly when frame counts must agree across
features. The spectral shape features also accept `frame_size=None` for
one periodic-Hann-windowed FFT of the whole clip.

`pitch` and `hnr` work on unpadded frames of the raw signal (`hnr` pads a
clip shorter than one frame with zeros); `thd` frames like the
time-domain features.

With `return_metadata=True` every extractor also returns
`metadata['times']`, the time in seconds of each frame's center in the
original signal (`librosa.frames_to_time(i, sr, hop_length)` for centered
frames, plus `frame_size // 2` samples otherwise), so features with the
same `frame_size` and `hop_length` share a time axis whatever `center` is.

## Aggregation

`extract_all_features` reduces each per-frame array over its last axis:

| `aggregation` | Output keys | Statistic |
|---|---|---|
| `mean` (default) | `name` | mean |
| `std` | `name` | standard deviation |
| `summary` | `name_mean`, `name_std` | both |
| `advanced_summary` | `name_mean`, `name_std`, `name_skew`, `name_kurt` | plus skewness and excess kurtosis |
| `skew`, `kurtosis` | `name` | one moment |
| `sliding_window` | `name` | `sliding_agg` (mean, std, max, min) over windows of `window_size` frames every `hop_length` frames, given under `aggregation_params` |
| `stack` | single vector | no reduction, everything concatenated |

`flatten` (default true) makes every aggregated array 1-D.
`preserve_shape: true` skips aggregation and returns the per-frame arrays.

`normalize` (`zscore` or `minmax`) rescales each aggregated feature by
the statistics of its own values in the same clip, for example across the
MFCC coefficients; single-value features are left as they are. With
`aggregation_params.norm_stats` (per-feature mean/std or min/max computed
over a dataset) every feature is scaled by those instead. For training,
leave it `null` and scale in scikit-learn.

## Undefined frames and coverage

A frame where a feature has no defined value is NaN: pitch on an unvoiced
frame, HNR, THD and spectral kurtosis on a zero-energy frame, kurtosis on
a constant frame, formants where fewer than `n_formants` resonances pass
the bandwidth test. `afx.feature_info(name)['undefined']` states the
condition for each feature (None for features that are always finite,
which give 0 or a log floor on silence). Every statistic skips NaN
frames, so `pitch_mean` is the mean f0 of the voiced frames; a statistic
is NaN only when every frame is undefined, as for the pitch of silence
or noise.

`"coverage": true` adds `<name>_valid_frac` after each feature's
statistics: the fraction of frames the statistics are based on. A frame
of a multi-row feature (MFCC, formants) counts as valid only when all of
its rows are finite, so `formants_valid_frac` is the fraction of frames
with all `n_formants` formants. With `sliding_window` it has one value
per window. It is never normalized, and it is an error with `stack` or
`preserve_shape`, which keep every frame. The same option is the
`coverage` argument of `extract_features_from_folder`,
`AFXFeatureExtractor` and `afx_audio.torch.AFXFeatureModule`, and
`afx --coverage` on the command line.

```python
cfg = {'features': {'pitch': {}, 'hnr': {}}, 'aggregation': 'summary', 'coverage': True}
afx.extract_all_features(x, sr, cfg)
# {'pitch_mean': ..., 'pitch_std': ..., 'pitch_valid_frac': array([0.48]), 'hnr_mean': ..., ...}
```

`afx.valid_fraction(per_frame)` gives the same number for an array from
`preserve_shape` or a single extractor, and
`afx_audio.utils.aggregator.valid_frame_mask` the per-frame mask.
`extract_pitch(..., return_metadata=True)` also returns
`metadata['voiced']`, the boolean voicing mask; the low-level
`afx_audio.utils.pitch.yin` keeps 0 for unvoiced frames.

## Input

Signals must be 1-D floating-point arrays without NaN or infinity;
integer PCM must be scaled by the caller (`x / 32768.0` for int16).
`afx.load_audio(path, sr)` returns mono float32 audio resampled with a
polyphase filter, or the file's own rate with `sr=None`; `offset` and
`duration` (seconds) read part of a file.

## Presets

`afx.list_presets()` returns `bioacoustics`, `egemaps`, `machinery`,
`music` and `speech`; `afx.load_preset(name)` returns a validated config and
`afx.preset_info(name)` its id, title, description, sample rate and
features. The JSON files are in `afx_audio/presets/`; copy one as a
starting point.

Presets are versioned. `'speech'` loads the latest version and
`'speech@1'` exactly version 1, here and in `afx --preset` and
`AFXFeatureExtractor(config=...)`; `afx --list-presets` and
`afx.list_presets(versions=True)` list every id. Version N is the file
`speech@N.json` and carries `"preset": {"name": "speech", "version": N}`.
A released version is never edited: `tests/test_presets.py` pins the
`config_hash` of each one, and a change ships as the next version while
the old file stays.

Batch and sklearn provenance records report `preset_id` (`'speech@1'`) and
`preset_modified`, True when an override changed a value the preset sets
(sample rate, features, aggregation). To cite an extraction, give the
package version, the preset id and the config hash from the record, for
example "features: afx-audio 0.3.0, preset speech@1, config hash
6a8de7dfed40". Parameters a preset leaves at their defaults follow the
package version, which is why both are needed.

| Preset | Sample rate | Contents |
|---|---|---|
| `speech` | 16 kHz | 20 MFCC with deltas, pitch, formants, HNR, energy, spectral contrast; `summary` |
| `bioacoustics` | 32 kHz | mel and MFCC, spectral shape, flatness, entropy |
| `music` | 22.05 kHz | chroma (STFT and CQT), tonnetz, onset strength, spectral contrast, MFCC |
| `machinery` | 16 kHz | frame-wise crest factor, kurtosis, Hjorth parameters, HNR; `advanced_summary` |
