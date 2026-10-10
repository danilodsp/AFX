# Notebooks

Install the dependencies with `pip install -e ".[notebooks]"` (add `torch` for the PyTorch notebook).

| Notebook | Content | Audio |
|---|---|---|
| `feature_benchmark_validation.ipynb` | AFX against librosa, torchaudio, openSMILE, spafe and Praat, with a PASS/FAIL tolerance report (`feature_benchmark_report.csv`, git-ignored) | synthetic |
| `domain_feature_packs_demo.ipynb` | each preset end to end: download a small open dataset, extract, train a classifier, plot the confusion matrix, importances and a PCA view (helpers in `domain_packs.py`) | downloads |
| `torch_features_demo.ipynb` | NumPy/torch parity, batched extraction, gradients, `AFXFeatureModule` as a front end | synthetic fallback |
| `sklearn_pipeline_demo.ipynb` | `AFXFeatureExtractor` in pipelines, cross-validation, `GridSearchCV` over extraction settings | synthetic fallback |
| `parallel_batch_extraction_demo.ipynb` | `n_jobs` batch extraction, serial vs parallel timing, Parquet output | synthetic fallback |
| `feature_modules_demo.ipynb` | one extractor at a time | dataset |
| `feature_aggregation_demo.ipynb` | frame-level vs clip-level aggregation | dataset |
| `feature_selection_demo.ipynb` | selecting feature subsets; reads `batch_features.npy` from the batch notebook | dataset |
| `feature_2d_visualization_demo.ipynb` | plotting 2-D features | ESC-50 |
| `batch_feature_extraction_demo.ipynb` | folder extraction with `extract_features_from_folder` | dataset |

## Datasets

Notebooks marked "dataset" read audio from the folder in one of these environment variables, searched recursively, so any folder of `.wav`, `.ogg` or `.flac` files works:

```bash
export ESC_50_dataset=/path/to/ESC-50          # folder with audio/ and meta/esc50.csv
export UrbanSound8K_dataset=/path/to/UrbanSound8K
export dataset_nonspeech7k=/path/to/Nonspeech7k  # or nonspeech7k_dataset
```

The three "synthetic fallback" notebooks synthesize clips when none is set.

## Domain packs

`domain_feature_packs_demo.ipynb` runs one preset per `DOMAIN` value:

| `DOMAIN` | Dataset | Task |
|---|---|---|
| `speech` | RAVDESS speech | emotion classification |
| `bioacoustics` | ESC-50, animal classes | animal sound classification |
| `music` | GTZAN | genre classification |
| `machinery` | DCASE 2020 Task 2 dev, pump | normal vs anomaly |

Datasets download into `$DATASETS` (default `./datasets`); with the default per-class subsets they stay under about 4 GB and each domain runs in a few minutes on a CPU:

```bash
DATASETS=/path/to/datasets jupyter nbconvert --to notebook --execute notebooks/domain_feature_packs_demo.ipynb
```

The benchmark notebook needs librosa; torchaudio, openSMILE, spafe and parselmouth sections are skipped when those packages are missing:

```bash
jupyter nbconvert --to notebook --execute notebooks/feature_benchmark_validation.ipynb
```
