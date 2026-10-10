"""
Dataset statistics: streaming accumulator, batch output and the StandardScaler built from it.
"""
import json

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from afx_audio import load_stats, read_provenance
from afx_audio.batch_extractor import extract_features_from_folder
from afx_audio.dataset_stats import DatasetStats

SR = 16000
CONFIG = {
    'sample_rate': SR,
    'features': {
        'zcr': {'frame_size': 1024, 'hop_length': 512},
        'rms_energy': {'frame_size': 1024, 'hop_length': 512},
        'mfcc': {'n_mfcc': 6, 'frame_size': 1024, 'hop_length': 512},
        'spectral_centroid': {'frame_size': 1024, 'hop_length': 512},
    },
    'aggregation': 'summary',
}


@pytest.fixture
def audio_folder(tmp_path):
    rng = np.random.default_rng(3)
    folder = tmp_path / 'audio'
    folder.mkdir()
    t = np.arange(SR // 4) / SR
    for i in range(7):
        sig = (0.2 + 0.1 * i) * np.sin(2 * np.pi * (150 + 90 * i) * t) + 0.05 * rng.standard_normal(t.size)
        sf.write(folder / f'clip_{i}.wav', sig.astype(np.float32), SR)
    return folder


def _matrix(parquet_path):
    return pd.read_parquet(parquet_path)


@pytest.mark.parametrize('n_jobs', [1, 2])
def test_batch_stats_equal_numpy(audio_folder, tmp_path, n_jobs):
    out = tmp_path / 'f.parquet'
    extract_features_from_folder(str(audio_folder), CONFIG, str(out), n_jobs=n_jobs,
                                 dataset_stats=True, verbose=False)
    df = _matrix(out)
    X = df.to_numpy()
    stats = load_stats(out)

    assert stats['columns'] == list(df.columns)
    assert stats['n_rows'] == 7
    np.testing.assert_array_equal(stats['count'], np.full(X.shape[1], 7))
    np.testing.assert_array_equal(stats['nan_count'], np.zeros(X.shape[1]))
    np.testing.assert_allclose(stats['mean'], np.nanmean(X, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats['var'], np.nanvar(X, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stats['std'], np.nanstd(X, axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(stats['min'], X.min(axis=0))
    np.testing.assert_array_equal(stats['max'], X.max(axis=0))
    assert stats['config_hash'] == read_provenance(out)['config_hash']


def test_serial_and_parallel_stats_files_identical(audio_folder, tmp_path):
    for name, n_jobs in (('s', 1), ('p', 2)):
        extract_features_from_folder(str(audio_folder), CONFIG, str(tmp_path / f'{name}.csv'),
                                     n_jobs=n_jobs, dataset_stats=True, verbose=False)
    assert (tmp_path / 's.csv.stats.json').read_text() == (tmp_path / 'p.csv.stats.json').read_text()


def test_stats_off_by_default(audio_folder, tmp_path):
    extract_features_from_folder(str(audio_folder), CONFIG, str(tmp_path / 'f.npy'), verbose=False)
    assert not (tmp_path / 'f.npy.stats.json').exists()
    with pytest.raises(FileNotFoundError, match='dataset_stats=True'):
        load_stats(tmp_path / 'f.npy')


def test_load_stats_accepts_both_paths(audio_folder, tmp_path):
    out = tmp_path / 'f.json'
    extract_features_from_folder(str(audio_folder), CONFIG, str(out), dataset_stats=True, verbose=False)
    a, b = load_stats(out), load_stats(tmp_path / 'f.json.stats.json')
    assert a['columns'] == b['columns']
    np.testing.assert_array_equal(a['mean'], b['mean'])


def _random_rows(rng, n_rows, n_cols, nan_fraction):
    X = rng.normal(5.0, 3.0, size=(n_rows, n_cols)) * rng.uniform(0.1, 1e3, size=n_cols)
    X[rng.random(X.shape) < nan_fraction] = np.nan
    return X


def _accumulate(names, X):
    stats = DatasetStats()
    for row in X:
        stats.update(names, row)
    return stats


def test_welford_and_chan_merge_match_numpy():
    rng = np.random.default_rng(0)
    X = _random_rows(rng, 200, 9, 0.1)
    X[:, 3] = np.nan
    X[:, 4] = 7.25
    names = [f'c{i}' for i in range(X.shape[1])]

    single = _accumulate(names, X).to_dict()
    merged = _accumulate(names, X[:13]).merge(_accumulate(names, X[13:120])).merge(
        _accumulate(names, X[120:])).to_dict()
    for stats in (single, merged):
        assert stats['n_rows'] == 200
        np.testing.assert_array_equal(stats['count'], np.sum(~np.isnan(X), axis=0))
        np.testing.assert_array_equal(stats['nan_count'], np.sum(np.isnan(X), axis=0))
        with np.errstate(invalid='ignore'), pytest.warns(RuntimeWarning):
            mean, var = np.nanmean(X, axis=0), np.nanvar(X, axis=0)
        np.testing.assert_allclose(stats['mean'], mean, rtol=1e-12)
        np.testing.assert_allclose(stats['var'], var, rtol=1e-12, atol=1e-12)
        assert np.isnan(stats['mean'][3]) and np.isnan(stats['min'][3])
        assert stats['var'][4] == 0.0
    np.testing.assert_allclose(merged['mean'], single['mean'], rtol=1e-12)
    np.testing.assert_allclose(merged['var'], single['var'], rtol=1e-12)


def test_merge_with_disjoint_and_new_columns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(10, 3))
    a = _accumulate(['x', 'y'], X[:4, :2])
    b = _accumulate(['y', 'z'], X[4:, 1:])
    stats = a.merge(b).to_dict()
    assert stats['columns'] == ['x', 'y', 'z']
    np.testing.assert_array_equal(stats['count'], [4, 10, 6])
    np.testing.assert_allclose(stats['mean'], [X[:4, 0].mean(), X[:, 1].mean(), X[4:, 2].mean()], rtol=1e-12)
    np.testing.assert_allclose(stats['var'], [X[:4, 0].var(), X[:, 1].var(), X[4:, 2].var()], rtol=1e-12)


def test_update_rejects_shape_mismatch():
    with pytest.raises(ValueError, match='one entry per name'):
        DatasetStats().update(['a', 'b'], np.zeros(3))


def test_all_nan_column_written_as_null(tmp_path):
    from afx_audio.dataset_stats import write_stats

    stats = _accumulate(['a', 'b'], np.array([[1.0, np.nan], [3.0, np.nan]])).to_dict()
    path = tmp_path / 'x.stats.json'
    write_stats(str(path), stats)
    record = json.loads(path.read_text())
    assert record['mean'] == [2.0, None]
    assert record['nan_count'] == [0, 2]
    loaded = load_stats(path)
    assert np.isnan(loaded['mean'][1]) and loaded['mean'][0] == 2.0


def test_scaler_from_stats_matches_standard_scaler():
    pytest.importorskip('sklearn')
    from sklearn.preprocessing import StandardScaler

    from afx_audio.sklearn import scaler_from_stats

    rng = np.random.default_rng(2)
    X = _random_rows(rng, 150, 6, 0.0)
    X[:, 2] = 4.0  # constant column: scale 1, as StandardScaler gives
    names = [f'c{i}' for i in range(X.shape[1])]
    ours = scaler_from_stats(_accumulate(names, X).to_dict())
    ref = StandardScaler().fit(X)

    np.testing.assert_allclose(ours.mean_, ref.mean_, rtol=1e-12)
    np.testing.assert_allclose(ours.var_, ref.var_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(ours.scale_, ref.scale_, rtol=1e-12)
    assert ours.n_samples_seen_ == ref.n_samples_seen_ == 150
    X_new = _random_rows(rng, 20, 6, 0.0)
    np.testing.assert_allclose(ours.transform(X_new), ref.transform(X_new), rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ours.inverse_transform(ours.transform(X_new)), X_new, rtol=1e-10)


def test_scaler_from_stats_with_nan_and_column_subset():
    pytest.importorskip('sklearn')
    from sklearn.preprocessing import StandardScaler

    from afx_audio.sklearn import scaler_from_stats

    rng = np.random.default_rng(4)
    X = _random_rows(rng, 80, 5, 0.15)
    X[:, 1] = np.nan
    names = [f'c{i}' for i in range(X.shape[1])]
    stats = _accumulate(names, X).to_dict()

    with pytest.raises(ValueError, match="no non-NaN value.*'c1'"):
        scaler_from_stats(stats)
    with pytest.raises(ValueError, match='not in the statistics'):
        scaler_from_stats(stats, columns=['c0', 'nope'])

    keep = ['c4', 'c0', 'c2', 'c3']
    ours = scaler_from_stats(stats, columns=keep)
    cols = [names.index(c) for c in keep]
    ref = StandardScaler().fit(X[:, cols])
    np.testing.assert_allclose(ours.mean_, ref.mean_, rtol=1e-12)
    np.testing.assert_allclose(ours.scale_, ref.scale_, rtol=1e-12)
    np.testing.assert_array_equal(ours.n_samples_seen_, ref.n_samples_seen_)
    X_new = _random_rows(rng, 10, 4, 0.2)
    np.testing.assert_allclose(ours.transform(X_new), ref.transform(X_new), rtol=1e-10, equal_nan=True)


def test_scaler_from_batch_output(audio_folder, tmp_path):
    pytest.importorskip('sklearn')
    from sklearn.preprocessing import StandardScaler

    from afx_audio.sklearn import scaler_from_stats

    out = tmp_path / 'f.parquet'
    extract_features_from_folder(str(audio_folder), CONFIG, str(out), dataset_stats=True, verbose=False)
    X = _matrix(out).to_numpy()
    ours = scaler_from_stats(load_stats(out))
    np.testing.assert_allclose(ours.transform(X), StandardScaler().fit_transform(X), rtol=1e-9, atol=1e-9)
