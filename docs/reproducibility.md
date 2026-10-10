# Reproducibility

## Provenance of batch outputs

Every file written by `extract_features_from_folder` (and the `afx` CLI)
records how it was made. Parquet outputs carry the record in the schema
metadata under the key `afx`. The npy, json and csv outputs get a sidecar
`<output>.provenance.json`, so their own layouts are unchanged.

```python
import afx_audio as afx

afx.extract_features_from_folder('clips/', 'config.json', 'features.parquet')
record = afx.read_provenance('features.parquet')
record['config_hash']   # 12 hex characters
record['file_sha256']   # {'a.wav': '9e1f...', ...}
```

| Key | Content |
|---|---|
| `afx_version` | afx-audio version that wrote the file |
| `config_hash` | `afx.config_hash(config)`, 12 hex characters |
| `preset_id`, `preset_modified` | `'speech@1'` when the config came from a preset (else `null`), and whether an override changed a value the preset sets (`null` for a preset that is not bundled) |
| `config` | the validated config after `features`, `aggregation`, `sample_rate` and the other overrides |
| `sample_rate` | rate every file was loaded at |
| `resampling` | method (`scipy.signal.resample_poly`), whether `sample_rate` overrode the config, the count of files per native rate and how many were resampled |
| `format`, `n_files`, `n_errors` | output format, files found, files that failed |
| `custom_features` | names of features from `register_feature` used by the config; their code is not hashed |
| `file_sha256` | sha256 of each audio file's bytes, keyed like the output rows; `null` with `hash_files=False` (`afx --no-hash`) |
| `versions` | afx_audio, numpy, scipy, soundfile, libsndfile and Python versions |
| `platform`, `created_utc` | `sys.platform` and the UTC creation time (ISO 8601) |

Files are hashed in the worker that extracts them, so the cost is one
extra read per file. On one machine, 40 three-second 22.05 kHz WAV files
with the bundled `speech` preset took 2.03 s with hashing and 1.88 s
without (best of three, serial).

From the shell, `afx` prints the config hash after a run (unless
`--quiet`), and `afx --provenance FILE` prints the record of an existing
output:

```console
$ afx clips/ --preset speech -o features.parquet -q
$ afx --provenance features.parquet
```

## Config hash

`afx.config_hash(config)` is the first 12 hex characters of the sha256 of
the config's canonical JSON (sorted keys, no whitespace, NaN rejected).
Key order does not change it, and numpy scalars hash like the Python
numbers they hold. Batch outputs hash the validated config they record,
so `afx.config_hash(record['config']) == record['config_hash']` holds for
every output and two outputs with equal hashes were extracted with the
same parameters. A raw config that relies on defaults (for example a
`features` list instead of a dict) hashes differently from its validated
form; compare against `record['config']` rather than the file on disk.

`AFXFeatureExtractor` exposes the same information after `fit` as
`provenance_`, with the scikit-learn version added and no file hashes,
since its inputs may be arrays.

## Dataset statistics

`dataset_stats=True` (`afx --stats`) accumulates per-column statistics
over the flattened feature vectors in the same pass as extraction and
writes them to `<output>.stats.json`. The columns are those of the csv and
parquet outputs (`mfcc_mean_0`, ...), whatever the output format.

| Key | Content |
|---|---|
| `columns` | column names, in output order |
| `n_rows` | files that produced features |
| `count`, `nan_count` | non-NaN and NaN values per column |
| `mean`, `var` | mean and population variance (ddof=0) of the non-NaN values |
| `min`, `max` | extremes of the non-NaN values |
| `afx_version`, `config_hash` | link to the provenance record |

Means and variances use Welford's single-value update. `DatasetStats.merge`
combines shards with the pairwise formula of Chan, Golub and LeVeque
(1979), so statistics of a dataset extracted in several runs can be
combined without the feature matrices:

```python
from afx_audio.dataset_stats import DatasetStats
```

Rows are accumulated in sorted file order after extraction, so `n_jobs=1`
and `n_jobs=2` write byte-identical statistics. Against `numpy.nanmean`
and `numpy.nanvar` on the stacked matrix the tests require 1e-12 relative
agreement; the 40-file run above (154 columns) agreed to 2.2e-14 for the
mean and 7.0e-14 for the variance. Values in `stats.json` that are NaN or
infinite (a column with no non-NaN value) are written as `null`.

`afx.load_stats(path)` takes the output or the `.stats.json` file and
returns the record with numpy arrays plus `std`.

## Normalizing new data with dataset statistics

`afx_audio.sklearn.scaler_from_stats(stats)` returns a fitted
`StandardScaler` equal to `StandardScaler().fit(X)` on the matrix the
statistics came from: same `mean_`, `var_`, `scale_` (scale 1 for
constant columns, by sklearn's rule) and `n_samples_seen_` (per-column
counts when values were missing). The training matrix never has to be
loaded.

```python
import pandas as pd
import afx_audio as afx
from afx_audio.sklearn import scaler_from_stats

afx.extract_features_from_folder('train/', 'config.json', 'train.parquet', dataset_stats=True)
scaler = scaler_from_stats(afx.load_stats('train.parquet'))

afx.extract_features_from_folder('test/', 'config.json', 'test.parquet')
X_test = scaler.transform(pd.read_parquet('test.parquet').to_numpy())
```

A column with no non-NaN value has neither a mean nor a scale, so
`scaler_from_stats` raises `ValueError` naming it. Pass `columns=` with
the columns to keep (in the order the data will have) and drop the same
columns from the data. Check that `afx.read_provenance('test.parquet')['config_hash']`
equals the training statistics' `config_hash` before applying the scaler.

## API

::: afx_audio.provenance.config_hash

::: afx_audio.provenance.read_provenance

::: afx_audio.dataset_stats.load_stats

::: afx_audio.dataset_stats.DatasetStats

::: afx_audio.sklearn.scaler_from_stats
