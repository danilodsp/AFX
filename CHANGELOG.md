# Changelog

## 0.3.0 (unreleased)

### Breaking changes

- The package is `afx-audio` on PyPI and imports as `afx_audio` (`import afx_audio as afx`); `import AFX` no longer works. The PyPI project `afx` is unrelated, and its `afx/` package collides with `AFX/` on case-insensitive file systems.
- Python 3.10 or newer is required.
- The core install is numpy, scipy and soundfile. scikit-learn, pandas and matplotlib moved to the `sklearn`, `parquet` and `plot` extras; `all` installs them with torch.
- Extractor arguments after `signal` and `sr` are keyword-only, as in librosa, in both `afx.extractors` and `afx.torch.functional`: `extract_mfcc(y, sr, n_mfcc=20)` works, `extract_mfcc(y, sr, 20)` raises. Several signatures gained parameters in the middle, so positional calls would otherwise have changed meaning silently.
- Unknown feature parameters raise instead of being ignored, both at config validation and when calling an extractor directly (`TypeError`). Non-positive frame sizes and counts are rejected, and so are unknown top-level config keys (`agregation`) and `sliding_window` without its window sizes.
- An unknown feature name raises `ValueError` (it was `KeyError` from `extract_all_features`), with the closest valid name in the message.
- scikit-learn 1.3 or newer for the `sklearn` extra.
- Signals must be 1-D floating point and finite. Integer PCM, complex, empty, multichannel and NaN/inf input raise with a message saying what to do, including through `AFXFeatureExtractor`.
- `spectral_rolloff` sums magnitudes and returns the bin frequency, as librosa does, instead of interpolating on the power spectrum (`power=2` gives an energy roll-off).
- `spectral_contrast` follows librosa: `power_to_db` on the peak and valley magnitudes (10 log10, top_db 80) and the quantile size taken from the untrimmed band. Values are half the previous ones.
- MFCC deltas use the Savitzky-Golay filter with fitted edges (`librosa.feature.delta`); `order=2` is the second derivative, not a delta of the delta.
- `zcr` divides by the frame length instead of frame length minus one and treats |x| <= 1e-10 as zero.
- `chroma_stft` uses librosa's chroma filterbank on the power spectrogram with max normalization. `f_ref` is replaced by `tuning`.
- Aggregation skips NaN frames: a statistic is NaN only when the feature is undefined on every frame.
- Sample entropy uses the same N - m templates for both lengths (Richman and Moorman), which changes values slightly.
- `extract_features_from_folder` infers the output format from the file extension (`save_format=None`), and rejects an unknown extension instead of writing a `.npy` file next to it. JSON output writes `null` for NaN.
- `afx` prints one-line `afx: error:` messages with exit status 2 for usage and configuration errors, and exits with 1 when no features were extracted. `--aggregation` no longer offers `sliding_window`, which needs window sizes from a config file.
- `framewise_extractor` takes `frame_size`, `hop_length`, `center` and `return_metadata` as keyword-only arguments and no longer forwards them to the wrapped function.
- `pitch` is NaN on unvoiced frames instead of 0, so pitch statistics cover voiced frames only and a clip with no voiced frame gives NaN `pitch_*`. `afx_audio.utils.pitch.yin` still returns 0.
- `from afx_audio import *` no longer exports `AFXFeatureExtractor`, so it works without scikit-learn; use `from afx_audio.sklearn import AFXFeatureExtractor` or `afx.AFXFeatureExtractor`.
- `AFXFeatureExtractor(features=[...])` orders output columns as `features`, and its `sample_rate` and `aggregation` default to None, meaning the config's values, instead of overriding a preset with 22050 Hz and `mean`.

### Fixed

- HNR used the biased autocorrelation, so r_max could not exceed (N - lag)/N: a pure tone read about 13 dB at 2048/22050 and about 2 dB with the speech preset framing. It now uses the normalized cross-correlation of the overlapping segments (NCCF) and tracks SNR within 1 dB on tone-plus-noise tests.
- `spectral_flatness` returned 0 or 1 for silent frames depending on rounding; it is 1, as in librosa.
- On the torch path, silent frames gave NaN gradients through `spectral_bandwidth` and `spectral_flux` (the derivative of sqrt at 0).
- `AFXFeatureModule(aggregation='mean_std')` concatenated along the batch axis for per-frame features (shape `(2B,)` instead of `(B, 2)`) and raised on unbatched input.
- Frame-time metadata added `frame_size // 2` for `center=True` and nothing for `center=False`, the reverse of the librosa convention. All extractors now report frame centers through `afx.utils.frame_utils.frame_times`, so features with the same framing share one time axis.
- `aggregation: "sliding_window"` could not be used from a config because `window_size` and `hop_length` were never forwarded; pass them under `aggregation_params`.
- Frame-wise time-domain features on a clip shorter than one frame were computed on the zero-padded frame. The clip is now analyzed as one unpadded frame.
- FFT round-off in the autocorrelation made `autocorr_zero_crossings` count random sign flips where the exact value is 0.
- Float32 input lost precision in kurtosis and energy ratios; time-domain moments are accumulated in float64.
- `spectral_flux` on a single frame returned an empty array; it returns `[0.0]`.
- The STFT cache was not safe to use from several threads.
- `examples/example_extract_and_plot.py` plotted the aggregated MFCC vector and always raised.
- The STFT result cache keyed entries on a full copy of the signal bytes; it now stores a BLAKE2b digest.
- `AFXFeatureExtractor` declares `__sklearn_tags__` for scikit-learn 1.6 and later.

### Added

- eGeMAPS v02 (Eyben et al., 2016) under MIT: `afx.egemaps(signal, sr)` returns the 88 functionals with openSMILE's names and order, `FEATURE_MAP` gains `egemaps` (88 values per clip) and `egemaps_lld` (25 descriptors per 10 ms frame), and there is an `egemaps` preset. Agreement with openSMILE per functional is in `docs/egemaps.md`, written by `benchmarks/validate_egemaps.py`.
- Provenance in batch outputs: config, `config_hash`, library versions, sample rate and resampling, and the sha256 of every input file. Parquet keeps it in the schema metadata key `afx`; other formats write `<output>.provenance.json`. `afx.config_hash`, `afx.read_provenance`, `AFXFeatureExtractor.provenance_`, `afx --provenance FILE` and `--no-hash`.
- Dataset statistics: `extract_features_from_folder(dataset_stats=True)` and `afx --stats` write per-column count, mean, variance, min, max and NaN count to `<output>.stats.json` in one streaming pass (Welford, with Chan's merge for workers); `afx.load_stats` and `afx_audio.sklearn.scaler_from_stats` turn it into a fitted `StandardScaler`.
- Coverage: `"coverage": true` (also `coverage=` on `extract_features_from_folder`, `AFXFeatureExtractor`, `AFXFeatureModule`, and `afx --coverage`) adds `<feature>_valid_frac`, the fraction of frames a feature's statistics are based on. `afx.valid_fraction`, `metadata['voiced']` from `extract_pitch`, and `feature_info(name)['undefined']`, which says when each feature is NaN.
- Numbered notebooks `01_quickstart` to `07_domain_presets` that run on synthetic audio without a dataset, `notebooks/demo_data.py`, and `tests/test_notebooks.py` (runs them with `AFX_RUN_NOTEBOOKS=1`).
- `benchmarks/bench_features.py --librosa --torch-batch N --markdown` and the `docs/benchmarks.md` page; `docs/reproducibility.md`; a JOSS paper draft in `paper/`.
- `afx_audio.methods.lpc.levinson`.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue and pull request templates, a pre-commit configuration, Dependabot, and coverage upload in CI.
- The sdist contains the tests, examples, docs sources, benchmarks and `CITATION.cff`, and the test suite passes from it. Releases install the built sdist and wheel in clean environments and run the tests before publishing.
- `center` for the time-domain, autocorrelation and THD extractors, giving the same frames and times as a centered STFT; centered `zcr` and `rms_energy` match librosa.
- `afx.list_features(group)`, `afx.feature_info(name)` (group, units, rows, parameters, definition) and `afx --describe FEATURE`.
- `tonnetz` and `onset_strength`, equal to librosa's `feature.tonnetz` (on chroma_stft) and `onset.onset_strength`.
- `AFXFeatureExtractor` accepts a preset name as `config` and a dict of per-feature parameters as `features`. `AFXFeatureModule` accepts `aggregation='summary'`, rejects integer tensors, and names its vector columns with `get_feature_names_out()`.
- `afx.extract_segments` returns one aggregated vector per fixed-length segment of a long signal.
- `afx.register_feature` and `unregister_feature` add custom extractors that configs, validation, `feature_info` and every front end accept.
- `afx.feature_names` and `afx.flatten_features` for the `name` / `name_i` columns used by every tabular output.
- `load_audio(path, sr, offset, duration)` reads part of a file and accepts file objects.
- A config's `features` may be a list of names.
- Parameters that were hidden in `**kwargs` are named: `mfcc` `top_db`; `mfcc_delta` `width`, `n_mels`, `fmin`, `fmax`; `pitch` `threshold`; `thd` `frame_size`, `hop_length`, `n_harmonics`; `chroma_stft` `tuning`, `normalize`; `cqt` and `chroma_cqt` `n_bins`, `bins_per_octave`, `fmin`.
- `afx.methods.chroma_filterbank` and `chroma_from_power`.
- `spectral_kurtosis`, `spectral_crest` and `spectral_decrease` (Peeters, 2004), on both the NumPy and torch paths.
- `lpc` (autocorrelation method, Levinson-Durbin over all frames at once) and `formants` (LPC root solving with bandwidth and frequency limits), with `afx.methods.lpc` and `lpc_roots`. The speech preset includes F1 to F3.
- The torch path covers 20 features: MFCC deltas, GFCC, chroma_stft, spectral skewness, kurtosis, slope, crest, decrease and contrast were added, `rms_energy` and `zcr` take framing and `center`, and `afx.torch.functional.delta` reproduces `compute_delta`.
- `tests/test_librosa_parity.py` (fourteen features against librosa), edge-case tests for every feature, a documentation site (`mkdocs`, `docs/`), a mypy job, `py.typed`, and `benchmarks/bench_features.py`.
- `melspectrogram`, `chroma_stft` and `cqt` accept `center`; `melspectrogram` accepts `fmin` and `fmax` as named parameters.
- `validate_config` (also run by `load_config`, `extract_all_features`, the batch extractor and the scikit-learn transformer) rejects unknown feature names and parameters, bad `aggregation` or `normalize` values and non-object feature parameters.
- `extract_features_from_folder` accepts a config dict, `sample_rate` and `verbose`.
- The batch extractor checks the output folder and the Parquet dependencies before extracting, and `--features` may name features that are not in the config (they get default parameters).
- CLI: `--preset`, `-o`, `--version`, `--sample-rate`, `--quiet`, `--list-features`, `--list-presets`, `--describe`; the output format follows the `--output` extension when `--format` is omitted.
- `afx.methods.autocorrelation` (FFT autocorrelation shared by the time-domain, pitch and HNR code), `afx.__version__`, ruff configuration, Windows and macOS CI jobs, wheel build check.

### Changed

- Documentation moved to https://danilopena.com/AFX/.
- Workflow actions are pinned to commit SHAs, with read-only default permissions.
- Extractor docstrings cite Davis and Mermelstein (MFCC), Brown (CQT bin layout), Boersma (HNR), Böck and Widmer (SuperFlux onset strength) and Shao and Wang (GFCC).
- The demo notebooks were replaced by the numbered set; `feature_benchmark_validation.ipynb` is now `benchmark_validation.ipynb`.
- Time-domain extractors, YIN, HNR, THD, chroma folding and sliding-window aggregation are vectorized over frames. On 10 s at 22.05 kHz: pitch 88 to 50 ms, hnr 77 to 45 ms, entropy 88 to 40 ms, autocorr_variance 66 to 20 ms, chroma_stft 48 to 21 ms.
- Sample entropy checks the candidate template pairs found by sorting directly when there are few of them (frames up to about 1500 samples of noise-like audio) and keeps the KD-tree count otherwise; results are the same.
- `import afx_audio as afx` takes about 0.4 s instead of 1.3 s (scipy submodules are imported where used).
- CSV output no longer needs pandas. Batch output keys use forward slashes on every OS, and failed or missing files are reported with warnings instead of prints.
- Whole-clip spectral features use the periodic Hann window, like the STFT.
- `extract_all_features` rejects non-1-D signals with a clear error.
- `frame_signal` is vectorized (stride tricks instead of a Python loop).
- Removed the unused window cache helpers and `pseudo_cqt` (use `cqt_approximation`), and the requirements files (pyproject.toml lists the dependencies).

## 0.2.0

Domain presets, scikit-learn transformer, PyTorch path, parallel batch extraction with Parquet output.
