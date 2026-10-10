# Notebooks

The [`notebooks/`](https://github.com/danilodsp/AFX/tree/main/notebooks) folder holds a numbered tutorial set. Each notebook runs on synthesized audio when no dataset is configured, so a fresh clone needs only `pip install "afx-audio[notebooks]"`.

| Notebook | Covers |
|---|---|
| [01 Quickstart](https://github.com/danilodsp/AFX/blob/main/notebooks/01_quickstart.ipynb) | `load_audio`, configs, `extract_all_features`, `flatten_features`, presets, the feature catalogue, config validation |
| [02 Frames and aggregation](https://github.com/danilodsp/AFX/blob/main/notebooks/02_frames_and_aggregation.ipynb) | per-frame arrays and frame times, plots, aggregation methods, NaN frames |
| [03 Long recordings and custom features](https://github.com/danilodsp/AFX/blob/main/notebooks/03_long_recordings_and_custom_features.ipynb) | `extract_segments`, `register_feature`, worker processes |
| [04 Batch and parallel](https://github.com/danilodsp/AFX/blob/main/notebooks/04_batch_and_parallel.ipynb) | `extract_features_from_folder`, output formats, failed files, `n_jobs`, the `afx` command |
| [05 scikit-learn and selection](https://github.com/danilodsp/AFX/blob/main/notebooks/05_sklearn_and_selection.ipynb) | `AFXFeatureExtractor`, cross-validation, grid search over extraction settings, feature selectors |
| [06 PyTorch](https://github.com/danilodsp/AFX/blob/main/notebooks/06_torch.ipynb) | `afx_audio.torch`: parity, batches, gradients, `AFXFeatureModule` |
| [07 Domain presets](https://github.com/danilodsp/AFX/blob/main/notebooks/07_domain_presets.ipynb) | a preset end to end on its domain dataset |
| [Benchmark validation](https://github.com/danilodsp/AFX/blob/main/notebooks/benchmark_validation.ipynb) | agreement with librosa, torchaudio, openSMILE, spafe and Praat |

Dataset environment variables and the test runner are described in [`notebooks/README.md`](https://github.com/danilodsp/AFX/blob/main/notebooks/README.md).
