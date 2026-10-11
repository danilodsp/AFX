# Notebooks

Run them in order; each one runs on synthesized audio when no dataset is configured. Install the extras with `pip install "afx-audio[notebooks]"` (plus `"afx-audio[torch]"` for 06).

| Notebook | Covers |
|---|---|
| `01_quickstart.ipynb` | `load_audio` (with `offset`/`duration`), keyword-only extractor parameters, configs, `extract_all_features`, `flatten_features`, preset ids (`speech@1`, `list_presets(versions=True)`, `preset_info`), `list_features`/`feature_info` with its `undefined` and `torch` entries, `validate_config` |
| `02_frames_and_aggregation.ipynb` | per-frame arrays (`preserve_shape`), frame times shared by every feature with `center=True` (`return_metadata`), plots with `afx_audio.utils.visualization`, NaN frames, `"coverage": true` and `<feature>_valid_frac` on a half-silent clip, every aggregation method, normalization |
| `03_long_recordings_and_custom_features.ipynb` | `extract_segments`, reading a window of a long file, `register_feature`/`unregister_feature`, picklable features for worker processes, the `__main__` guard |
| `04_batch_and_parallel.ipynb` | `extract_features_from_folder`, output formats, `errors.json`, filtering `AFXWarning`, provenance (`read_provenance`, `config_hash`, file hashes, `afx --provenance`), dataset statistics (`dataset_stats=True`, `--stats`, `load_stats`, `scaler_from_stats`), serial vs `n_jobs`, the `afx` command |
| `05_sklearn_and_selection.ipynb` | `AFXFeatureExtractor` in a Pipeline, `config='speech@1'` and `provenance_`, cross-validation, `GridSearchCV` over extraction settings, `get_feature_names_out`, `afx_audio.utils.selectors` |
| `06_torch.ipynb` | `afx_audio.torch`: parity with NumPy, batches, gradients, `AFXFeatureModule` in a model |
| `07_domain_presets.ipynb` | a preset end to end on its domain dataset (`speech@1`, `bioacoustics@1`, `music@1`, `machinery@1`), ending with the preset id and config hash behind the result |
| `08_egemaps.ipynb` | eGeMAPS v02 on synthetic vowels with known F0, jitter and shimmer: the 88 functionals by name (`afx.egemaps`, `EGEMAPS_NAMES`), the 25 LLDs over time (`egemaps_lld`), the `egemaps@1` preset in a pipeline, an optional openSMILE comparison |
| `benchmark_validation.ipynb` | numerical agreement with librosa, torchaudio, openSMILE, spafe and Praat, with PASS/FAIL tolerances |

`demo_data.py` (dataset discovery, synthetic clips and synthetic voices) and `domain_packs.py` (dataset downloads for 07) sit next to the notebooks and are imported by them. Notebooks write only to temporary folders.

## Datasets

01 to 06 use a local environmental-sound dataset when one of these points at its root, and synthetic tones, sweeps, noise bursts and AM/FM tones otherwise:

```bash
export ESC_50_dataset=/path/to/ESC-50
export UrbanSound8K_dataset=/path/to/UrbanSound8K
export dataset_nonspeech7k=/path/to/Nonspeech7k
```

The first one set wins. Labels come from the dataset's metadata CSV.

07 downloads RAVDESS (210 MB), ESC-50 (600 MB), GTZAN (1.2 GB) or the DCASE 2020 pump set (1 GB) into `$DATASETS` on first use. Without `DATASETS`, with `AFX_SYNTHETIC=1`, or when the download fails, it runs on synthetic clips.

08 always runs on synthetic voices. Its openSMILE cell runs when `opensmile` is installed and is skipped otherwise; openSMILE's licence allows research use only, so it is not an AFX dependency.

`benchmark_validation.ipynb` needs `librosa`; its torchaudio, openSMILE, spafe and Praat sections skip when those packages are missing:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install opensmile spafe praat-parselmouth
```

## Running them as tests

```bash
AFX_RUN_NOTEBOOKS=1 python -m pytest tests/test_notebooks.py
```

executes 01 to 08 on synthetic data with a time budget per notebook.
