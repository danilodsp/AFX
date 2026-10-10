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

Frames where a feature is undefined are NaN (HNR of a silent frame,
kurtosis of a constant frame) and are skipped by every statistic; a
statistic is NaN only if the feature is undefined on all frames. Pitch
uses 0, not NaN, for unvoiced frames, so its mean includes them.

`flatten` (default true) makes every aggregated array 1-D.
`preserve_shape: true` skips aggregation and returns the per-frame arrays.

`normalize` (`zscore` or `minmax`) rescales each aggregated feature by
the statistics of its own values in the same clip, for example across the
MFCC coefficients; single-value features are left as they are. With
`aggregation_params.norm_stats` (per-feature mean/std or min/max computed
over a dataset) every feature is scaled by those instead. For training,
leave it `null` and scale in scikit-learn.

## Input

Signals must be 1-D floating-point arrays without NaN or infinity;
integer PCM must be scaled by the caller (`x / 32768.0` for int16).
`afx.load_audio(path, sr)` returns mono float32 audio resampled with a
polyphase filter, or the file's own rate with `sr=None`; `offset` and
`duration` (seconds) read part of a file.

## Presets

`afx.list_presets()` returns `bioacoustics`, `machinery`, `music` and
`speech`; `afx.load_preset(name)` returns a validated config and
`afx.preset_info(name)` its description. The JSON files are in
`afx_audio/presets/`; copy one as a starting point.

| Preset | Sample rate | Contents |
|---|---|---|
| `speech` | 16 kHz | 20 MFCC with deltas, pitch, formants, HNR, energy, spectral contrast; `summary` |
| `bioacoustics` | 32 kHz | mel and MFCC, spectral shape, flatness, entropy |
| `music` | 22.05 kHz | chroma (STFT and CQT), tonnetz, onset strength, spectral contrast, MFCC |
| `machinery` | 16 kHz | frame-wise crest factor, kurtosis, Hjorth parameters, HNR; `advanced_summary` |
