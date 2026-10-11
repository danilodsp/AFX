# ESC-50 baselines

Accuracy of two fixed classifiers on the features of each bundled preset,
so a preset id can be cited with a reference number next to it. The data
is ESC-50 (Piczak, 2015): 2000 clips of 5 s, 50 classes of 40 clips,
scored with the five official cross-validation folds (train on four, test
on the fifth, every clip tested once).

Produced by

```bash
python benchmarks/baselines.py ESC50_ROOT --write-docs
```

where `ESC50_ROOT` is the unpacked
[GitHub release](https://github.com/karolpiczak/ESC-50) (`audio/`,
`meta/esc50.csv`) or the
[Harvard Dataverse release](https://doi.org/10.7910/DVN/YDEPUT)
(`ESC-50.zip`, one folder per class). The numbers below come from the
Dataverse `ESC-50.zip` (OGG Vorbis, mono, 1959 clips at 44.1 kHz and 41 at
48 kHz, sha256 `f3cff358799882869cee8ff27ac0eb74d904f4bb0e308f5d8ee4ef5a3c2cd711`);
the fold is the first field of each file name, which is the `fold` column
of the GitHub metadata.

Each config is extracted with `extract_features_from_folder` at its own
`sample_rate` (resampled with `scipy.signal.resample_poly`),
with `dataset_stats=True` and Parquet output, and the preset is not
modified: the config hash in the table is the one pinned in
`tests/test_presets.py` and recorded in the Parquet provenance. Both
classifiers share the front end `SimpleImputer(strategy='median')` and
`StandardScaler()`, fitted on the training folds only:
`LogisticRegression(C=1.0, max_iter=5000)` and
`RandomForestClassifier(n_estimators=500, random_state=0)`. Accuracy is the
mean over the five folds, ± the standard deviation over folds (ddof 0).
"NaN columns" counts the columns with at least one NaN value over the 2000
clips, which the imputer fills with the training median. Extraction time is
wall time for all 2000 clips including file hashing.

<!-- results start -->
| Config | Config hash | Sample rate | Columns | NaN columns | Extraction (s) | Logistic regression | Random forest |
|---|---|---:|---:|---:|---:|---:|---:|
| `speech@1` | `71f5a1447dc2` | 16000 | 154 | 2 (1%) | 184 | 0.583 ± 0.039 | 0.612 ± 0.020 |
| `bioacoustics@1` | `f2fbc8a2bb11` | 32000 | 194 | 0 (0%) | 121 | 0.548 ± 0.028 | 0.548 ± 0.034 |
| `music@1` | `cef990b7ffe1` | 22050 | 114 | 0 (0%) | 167 | 0.617 ± 0.022 | 0.605 ± 0.027 |
| `machinery@1` | `563073e5d310` | 16000 | 96 | 0 (0%) | 100 | 0.556 ± 0.030 | 0.541 ± 0.028 |
| `egemaps@1` | `cd530990145e` | 16000 | 88 | 0 (0%) | 196 | 0.489 ± 0.020 | 0.514 ± 0.023 |
| `config.json` | `867dfd4b73ab` | 22050 | 343 | 2 (1%) | 669 | 0.477 ± 0.038 | 0.536 ± 0.030 |
<!-- results end -->

<!-- environment start -->
Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 logical CPUs, Linux 6.18.44-fc-v114; extraction with `n_jobs=2`.
Versions: afx-audio 0.3.0, Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, soundfile 0.14.0 (libsndfile 1.2.2), scikit-learn 1.9.1, pandas 3.0.6, pyarrow 26.0.0.
Clips: 2000 scored, 0 missing; native sample rates (44100 Hz: 1959, 48000 Hz: 41), resampled with `scipy.signal.resample_poly` to each config's rate.
<!-- environment end -->

## Reading the numbers

All six configs score between 0.48 and 0.62. That is above the 0.443 that
Piczak (2015) reports for a random forest on MFCC and zero-crossing rate
statistics, and below his two-layer CNN on log-mel patches (0.645, Piczak,
MLSP 2015), the 0.813 of human listeners in the dataset paper, and the 0.956
of AST pretrained on AudioSet (Gong et al., 2021). These are linear and
random-forest classifiers on clip-level statistics of hand-crafted features,
with default hyperparameters and no augmentation; they exist to compare
presets with each other and to show when a change to AFX moves them, not to
compete with learned representations. The fold standard deviation is 0.020
to 0.039, so differences under about 0.03 are not resolved by five folds, for
example `machinery@1` against `bioacoustics@1` with the logistic regression,
or `speech@1` against `music@1` and `bioacoustics@1` against `machinery@1`
with the random forest. `egemaps@1` is lower,
which fits a set of 88 functionals built around voiced speech (F0, jitter,
shimmer, formants) applied to environmental sound. `config.json` aggregates
with the mean only, so its 343 columns hold no within-clip spread; the
logistic regression reaches 0.477 on it and the random forest 0.536. The NaN
columns are `pitch` statistics of clips without a voiced frame (5 clips for
`speech@1`, 280 for `config.json`) and F3 (`formants_2`) in 6 clips for
`config.json`.

## Per fold

<!-- folds start -->
| Config | Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
|---|---|---:|---:|---:|---:|---:|
| `speech@1` | Logistic regression | 0.568 | 0.532 | 0.580 | 0.652 | 0.580 |
| `speech@1` | Random forest | 0.588 | 0.598 | 0.645 | 0.623 | 0.608 |
| `bioacoustics@1` | Logistic regression | 0.525 | 0.530 | 0.562 | 0.598 | 0.527 |
| `bioacoustics@1` | Random forest | 0.557 | 0.505 | 0.583 | 0.585 | 0.510 |
| `music@1` | Logistic regression | 0.627 | 0.623 | 0.600 | 0.650 | 0.588 |
| `music@1` | Random forest | 0.598 | 0.610 | 0.603 | 0.650 | 0.568 |
| `machinery@1` | Logistic regression | 0.542 | 0.540 | 0.557 | 0.613 | 0.527 |
| `machinery@1` | Random forest | 0.562 | 0.527 | 0.560 | 0.562 | 0.492 |
| `egemaps@1` | Logistic regression | 0.505 | 0.492 | 0.455 | 0.510 | 0.482 |
| `egemaps@1` | Random forest | 0.507 | 0.490 | 0.510 | 0.557 | 0.505 |
| `config.json` | Logistic regression | 0.477 | 0.448 | 0.510 | 0.527 | 0.425 |
| `config.json` | Random forest | 0.517 | 0.490 | 0.557 | 0.578 | 0.537 |
<!-- folds end -->

## References

K. J. Piczak. ESC: Dataset for Environmental Sound Classification. Proc. ACM
Multimedia, 2015. [doi:10.1145/2733373.2806390](https://doi.org/10.1145/2733373.2806390)

K. J. Piczak. Environmental sound classification with convolutional neural
networks. IEEE MLSP, 2015.
[doi:10.1109/MLSP.2015.7324337](https://doi.org/10.1109/MLSP.2015.7324337)

Y. Gong, Y.-A. Chung, J. Glass. AST: Audio Spectrogram Transformer.
Interspeech, 2021. [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
