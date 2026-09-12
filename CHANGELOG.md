# Changelog

## Unreleased

Fixed

- HNR used the biased autocorrelation, so r_max could not exceed (N - lag)/N: a pure tone read about 13 dB at 2048/22050 and about 2 dB with the speech preset framing. It now uses the normalized cross-correlation of the overlapping segments (NCCF) and tracks SNR within 1 dB on tone-plus-noise tests.
- `AFXFeatureModule(aggregation='mean_std')` concatenated along the batch axis for per-frame features (shape `(2B,)` instead of `(B, 2)`) and raised on unbatched input.
- Frame-time metadata added `frame_size // 2` for `center=True` and nothing for `center=False`, the reverse of the librosa convention. All extractors now report frame centers through `AFX.utils.frame_utils.frame_times`, so features with the same framing share one time axis.
- `aggregation: "sliding_window"` could not be used from a config because `window_size` and `hop_length` were never forwarded; pass them under `aggregation_params`.
- Frame-wise time-domain features on a clip shorter than one frame were computed on the zero-padded frame. The clip is now analyzed as one unpadded frame.
- `examples/example_extract_and_plot.py` plotted the aggregated MFCC vector and always raised.
- The STFT result cache keyed entries on a full copy of the signal bytes; it now stores a BLAKE2b digest.
- `AFXFeatureExtractor` declares `__sklearn_tags__` for scikit-learn 1.6 and later.

Added

- `melspectrogram`, `chroma_stft` and `cqt` accept `center`; `melspectrogram` accepts `fmin` and `fmax` as named parameters.
- `validate_config` (also run by `load_config`) rejects unknown feature names, bad `aggregation` or `normalize` values and non-object feature parameters.
- `extract_features_from_folder` accepts a config dict, `sample_rate` and `verbose`.
- CLI: `--preset`, `--sample-rate`, `--quiet`, `--list-features`, `--list-presets`; the output format follows the `--output` extension when `--format` is omitted.
- `AFX.methods.autocorrelation` (FFT autocorrelation shared by the time-domain, pitch and HNR code), `AFX.__version__`, `AFX[dev]` extra, ruff configuration, Windows and macOS CI jobs, wheel build check.

Changed

- `extract_all_features` rejects non-1-D signals with a clear error.
- `frame_signal` is vectorized (stride tricks instead of a Python loop).
- Removed the unused window cache helpers and `pseudo_cqt` (use `cqt_approximation`).

## 0.2.0

Domain presets, scikit-learn transformer, PyTorch path, parallel batch extraction with Parquet output.
