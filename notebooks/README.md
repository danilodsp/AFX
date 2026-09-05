# Jupyter notebooks for tutorials and examples

## Notebooks

- **`feature_benchmark_validation.ipynb`** — numerical validation of AFX
  features against `librosa`, `torchaudio`, and `openSMILE` with explicit
  tolerance / PASS-FAIL reporting. Run it to confirm AFX reproduces the standard
  feature definitions before adopting the library.
- **`domain_feature_packs_demo.ipynb`** — curated **domain presets**
  (`speech`, `bioacoustics`, `music`, `machinery`) demoed end-to-end. Set
  `DOMAIN` in the config cell and Run All: it downloads a small open dataset,
  extracts the preset's features, trains a scikit-learn classifier, and plots a
  confusion matrix, feature importances, and a PCA view. Helper logic lives in
  `domain_packs.py`. See the domain packs section below.
- **`torch_features_demo.ipynb`** — the differentiable PyTorch path
  (`AFX.torch`): NumPy/torch parity, batched extraction, gradient flow
  through the transforms, and `AFXFeatureModule` as a trainable front end.
  Requires `pip install "AFX[torch]"`.
- **`sklearn_pipeline_demo.ipynb`** — `AFX.sklearn.AFXFeatureExtractor` in
  sklearn Pipelines: cross-validation, `GridSearchCV` over extraction
  settings, and readable feature importances via `get_feature_names_out()`.
- **`parallel_batch_extraction_demo.ipynb`** — multiprocessing batch
  extraction (`n_jobs`) with serial-vs-parallel timing and Parquet output
  loaded back into pandas.
- `feature_modules_demo.ipynb` — per-extractor walkthrough.
- `feature_aggregation_demo.ipynb` — frame- vs file-level aggregation.
- `feature_selection_demo.ipynb` — selecting subsets of features.
- `feature_2d_visualization_demo.ipynb` — visualizing 2-D features.
- `batch_feature_extraction_demo.ipynb` — folder-based batch extraction.

### Audio datasets via environment variables

Notebooks that need real audio resolve an environmental-sound dataset from
environment variables — set any ONE of these to a local dataset root before
launching Jupyter:

```bash
export dataset_nonspeech7k=/path/to/Nonspeech7k    # or nonspeech7k_dataset
export ESC_50_dataset=/path/to/ESC-50
export UrbanSound8K_dataset=/path/to/UrbanSound8K
```

Audio files are discovered recursively, so any of the datasets above (or a
plain folder of `.wav`/`.ogg`/`.flac` files) works. The three demo notebooks
for the ML integrations (`torch_features_demo`, `sklearn_pipeline_demo`,
`parallel_batch_extraction_demo`) additionally fall back to synthesized
clips when no dataset is configured, so they run out of the box.

### Running the benchmark notebook

`librosa` is required; `torchaudio` and `openSMILE` are optional (their sections
skip cleanly if not installed):

```bash
pip install -e ".[notebooks]"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu  # optional
pip install opensmile                                                          # optional
jupyter nbconvert --to notebook --execute notebooks/feature_benchmark_validation.ipynb
```

The notebook writes a `feature_benchmark_report.csv` tolerance report next to
itself (git-ignored).

### Domain feature packs

`domain_feature_packs_demo.ipynb` showcases the curated presets shipped in
`AFX.presets`. Each pack is a small config tuned for one domain and wired to an
openly-licensed dataset and a mini task:

| `DOMAIN`       | Preset                            | Dataset                       | Task                         |
|----------------|-----------------------------------|-------------------------------|------------------------------|
| `speech`       | Speech & paralinguistics          | RAVDESS (speech audio)        | emotion classification       |
| `bioacoustics` | Bioacoustics                      | ESC-50 (animal subset)        | animal-sound classification  |
| `music`        | Music & MIR                       | GTZAN                         | genre classification         |
| `machinery`    | Machinery & condition monitoring  | DCASE 2020 Task 2 dev (pump)  | normal vs. anomaly detection |

Load a pack directly in code:

```python
from AFX.presets import list_presets, load_preset, preset_info
list_presets()                 # ['bioacoustics', 'machinery', 'music', 'speech']
config = load_preset("speech") # a validated AFX config dict, ready to use
```

Datasets download on demand into `$DATASETS` (falling back to `./datasets`).
With the default per-class subsets the total footprint stays under ~4 GB and
each domain runs in a few minutes on CPU. Run it headless with:

```bash
pip install -e ".[notebooks]"
DATASETS=/path/to/datasets jupyter nbconvert --to notebook --execute \
    notebooks/domain_feature_packs_demo.ipynb
```
