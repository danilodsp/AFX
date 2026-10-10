"""
Tests for parallel batch extraction and parquet output in
afx_audio.batch_extractor.
"""
import json
import os

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from afx_audio.batch_extractor import extract_features_from_folder

SR = 16000
RNG = np.random.default_rng(7)


@pytest.fixture
def audio_folder(tmp_path):
    folder = tmp_path / 'audio'
    (folder / 'sub').mkdir(parents=True)
    t = np.arange(SR // 2) / SR
    for i in range(4):
        sig = np.sin(2 * np.pi * (220 + 110 * i) * t) \
            + 0.02 * RNG.standard_normal(t.size)
        name = f'tone_{i}.wav' if i < 3 else os.path.join('sub',
                                                          'tone_3.wav')
        sf.write(folder / name, sig.astype(np.float32), SR)
    return folder


@pytest.fixture
def config_path(tmp_path):
    config = {
        'sample_rate': SR,
        'features': {
            'mfcc': {'n_mfcc': 13, 'frame_size': 1024, 'hop_length': 512},
            'zcr': {'frame_size': 1024, 'hop_length': 512},
            'spectral_centroid': {'frame_size': 1024, 'hop_length': 512},
        },
        'aggregation': 'mean',
    }
    p = tmp_path / 'config.json'
    p.write_text(json.dumps(config))
    return str(p)


def test_parallel_matches_serial(audio_folder, config_path, tmp_path):
    out_serial = str(tmp_path / 'serial.json')
    out_parallel = str(tmp_path / 'parallel.json')

    res_serial = extract_features_from_folder(
        str(audio_folder), config_path, out_serial, save_format='json'
    )
    res_parallel = extract_features_from_folder(
        str(audio_folder), config_path, out_parallel, save_format='json',
        n_jobs=2,
    )

    assert list(res_serial) == list(res_parallel)  # same keys, same order
    for key in res_serial:
        for feat in res_serial[key]:
            np.testing.assert_allclose(
                res_serial[key][feat], res_parallel[key][feat], rtol=1e-12
            )
    # Saved files identical too
    with open(out_serial) as f1, open(out_parallel) as f2:
        assert json.load(f1) == json.load(f2)


def test_n_jobs_zero_uses_all_cores(audio_folder, config_path, tmp_path):
    out = str(tmp_path / 'all_cores.json')
    res = extract_features_from_folder(
        str(audio_folder), config_path, out, save_format='json', n_jobs=0
    )
    assert len(res) == 4


def test_parquet_output(audio_folder, config_path, tmp_path):
    pytest.importorskip('pyarrow')
    out = str(tmp_path / 'features.parquet')
    res = extract_features_from_folder(
        str(audio_folder), config_path, out, save_format='parquet', n_jobs=2
    )
    df = pd.read_parquet(out)
    assert df.shape[0] == 4
    assert df.index.name == 'file'
    assert 'mfcc_0' in df.columns and 'zcr' in df.columns
    # Values round-trip vs the in-memory result
    first_key = df.index[0]
    np.testing.assert_allclose(
        df.loc[first_key, 'zcr'],
        np.asarray(res[first_key]['zcr']).ravel()[0],
        rtol=1e-9,
    )


def test_parquet_matches_csv(audio_folder, config_path, tmp_path):
    pytest.importorskip('pyarrow')
    out_pq = str(tmp_path / 'f.parquet')
    out_csv = str(tmp_path / 'f.csv')
    extract_features_from_folder(
        str(audio_folder), config_path, out_pq, save_format='parquet'
    )
    extract_features_from_folder(
        str(audio_folder), config_path, out_csv, save_format='csv'
    )
    df_pq = pd.read_parquet(out_pq)
    df_csv = pd.read_csv(out_csv, index_col='file')
    pd.testing.assert_frame_equal(
        df_pq, df_csv, check_exact=False, rtol=1e-9
    )


def test_subfolder_keys_preserved(audio_folder, config_path, tmp_path):
    out = str(tmp_path / 'keys.json')
    res = extract_features_from_folder(
        str(audio_folder), config_path, out, save_format='json', n_jobs=2
    )
    assert any(k.startswith('sub') for k in res)
