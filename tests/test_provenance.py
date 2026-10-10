"""
Provenance records of batch outputs, config hashing and the sklearn provenance_ attribute.
"""
import hashlib
import json
import os

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import soundfile as sf

import afx_audio
from afx_audio import cli, config_hash, read_provenance
from afx_audio.batch_extractor import extract_features_from_folder

SR = 16000
CONFIG = {
    'sample_rate': SR,
    'features': {
        'zcr': {'frame_size': 1024, 'hop_length': 512},
        'mfcc': {'n_mfcc': 4, 'frame_size': 1024, 'hop_length': 512},
    },
    'aggregation': 'mean',
}


@pytest.fixture
def audio_folder(tmp_path):
    folder = tmp_path / 'audio'
    (folder / 'sub').mkdir(parents=True)
    t = np.arange(SR // 4) / SR
    sf.write(folder / 'a.wav', np.sin(2 * np.pi * 300 * t).astype(np.float32), SR)
    sf.write(folder / 'sub' / 'b.wav', np.sin(2 * np.pi * 500 * t).astype(np.float32), SR)
    # A different native rate, so one file is resampled
    t8 = np.arange(8000 // 4) / 8000
    sf.write(folder / 'c.wav', np.sin(2 * np.pi * 200 * t8).astype(np.float32), 8000)
    return folder


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize('fmt', ['npy', 'json', 'csv', 'parquet'])
def test_provenance_roundtrip_every_format(audio_folder, tmp_path, fmt):
    out = tmp_path / f'features.{fmt}'
    extract_features_from_folder(str(audio_folder), CONFIG, str(out), verbose=False)
    record = read_provenance(out)

    assert record['afx_version'] == afx_audio.__version__
    assert record['format'] == fmt
    assert record['config_hash'] == config_hash(record['config'])
    assert len(record['config_hash']) == 12
    assert record['config']['features'] == CONFIG['features']
    assert record['sample_rate'] == SR
    assert record['resampling']['native_sample_rates'] == {'8000': 1, '16000': 2}
    assert record['resampling']['n_resampled'] == 1
    assert record['resampling']['sample_rate_overridden'] is False
    assert record['versions']['numpy'] == np.__version__
    assert set(record['versions']) >= {'afx_audio', 'numpy', 'scipy', 'python', 'soundfile'}
    assert record['created_utc'].endswith('+00:00')
    assert record['n_files'] == 3 and record['n_errors'] == 0
    assert record['file_sha256'] == {
        'a.wav': _sha256(audio_folder / 'a.wav'),
        'c.wav': _sha256(audio_folder / 'c.wav'),
        'sub/b.wav': _sha256(audio_folder / 'sub' / 'b.wav'),
    }

    sidecar = tmp_path / f'features.{fmt}.provenance.json'
    if fmt == 'parquet':
        assert not sidecar.exists()
        assert b'afx' in pq.read_schema(out).metadata
    else:
        assert read_provenance(sidecar) == record


def test_output_layouts_unchanged(audio_folder, tmp_path):
    extract_features_from_folder(str(audio_folder), CONFIG, str(tmp_path / 'f.json'), verbose=False)
    with open(tmp_path / 'f.json', encoding='utf-8') as f:
        assert set(json.load(f)) == {'a.wav', 'c.wav', 'sub/b.wav'}

    extract_features_from_folder(str(audio_folder), CONFIG, str(tmp_path / 'f.parquet'), verbose=False)
    df = pd.read_parquet(tmp_path / 'f.parquet')
    assert df.index.name == 'file'
    assert list(df.index) == ['a.wav', 'c.wav', 'sub/b.wav']
    assert df.shape == (3, 5)


def test_provenance_records_overrides_and_skips_hashes(audio_folder, tmp_path):
    out = tmp_path / 'f.csv'
    extract_features_from_folder(str(audio_folder), CONFIG, str(out), sample_rate=8000,
                                 features=['zcr'], hash_files=False, verbose=False)
    record = read_provenance(out)
    assert record['file_sha256'] is None
    assert record['sample_rate'] == 8000
    assert record['config']['sample_rate'] == 8000
    assert list(record['config']['features']) == ['zcr']
    assert record['resampling'] == {
        'method': 'scipy.signal.resample_poly',
        'sample_rate_overridden': True,
        'native_sample_rates': {'8000': 1, '16000': 2},
        'n_resampled': 2,
    }
    assert record['config_hash'] != config_hash(CONFIG)


def test_provenance_parallel_matches_serial(audio_folder, tmp_path):
    serial = read_provenance(_run(audio_folder, tmp_path / 's.json', n_jobs=1))
    parallel = read_provenance(_run(audio_folder, tmp_path / 'p.json', n_jobs=2))
    for record in (serial, parallel):
        del record['created_utc']
    assert serial == parallel


def _run(folder, out, **kwargs):
    extract_features_from_folder(str(folder), CONFIG, str(out), verbose=False, **kwargs)
    return out


def test_provenance_counts_failed_files(audio_folder, tmp_path):
    (audio_folder / 'broken.wav').write_bytes(b'not audio')
    out = tmp_path / 'f.json'
    with pytest.warns(UserWarning, match='failed'):
        extract_features_from_folder(str(audio_folder), CONFIG, str(out), verbose=False)
    record = read_provenance(out)
    assert record['n_files'] == 4 and record['n_errors'] == 1
    assert 'broken.wav' not in record['file_sha256']


def test_config_hash_ignores_key_order():
    reordered = {
        'aggregation': 'mean',
        'features': {
            'mfcc': {'hop_length': 512, 'frame_size': 1024, 'n_mfcc': 4},
            'zcr': {'hop_length': 512, 'frame_size': 1024},
        },
        'sample_rate': SR,
    }
    assert config_hash(reordered) == config_hash(CONFIG)
    assert config_hash({**CONFIG, 'aggregation': 'std'}) != config_hash(CONFIG)
    # numpy scalars hash like the Python numbers they hold
    assert config_hash({**CONFIG, 'sample_rate': np.int64(SR)}) == config_hash(CONFIG)


def test_config_hash_is_pinned():
    # A change here silently breaks the link between old outputs and their configs
    canonical = (b'{"aggregation":"mean","aggregation_params":{},"coverage":false,'
                 b'"features":{"zcr":{"center":true,"frame_size":2048,"hop_length":512,"threshold":1e-10}},'
                 b'"normalize":null,"preserve_shape":false,"sample_rate":16000}')
    assert config_hash({'sample_rate': 16000, 'features': {'zcr': {}}}) == hashlib.sha256(canonical).hexdigest()[:12]


def test_config_hash_ignores_defaults_labels_and_layout():
    plain = {'sample_rate': 16000, 'features': ['zcr']}
    spelled = {'sample_rate': 16000, 'title': 'x', 'flatten': False, 'aggregation': 'mean',
               'features': {'zcr': {'center': True, 'threshold': 1e-10}}}
    assert config_hash(plain) == config_hash(spelled)
    assert config_hash(plain) != config_hash({**plain, 'features': {'zcr': {'center': False}}})


def test_config_hash_rejects_bad_input():
    with pytest.raises(TypeError, match='dict'):
        config_hash(['mfcc'])
    with pytest.raises(ValueError):
        config_hash({'sample_rate': float('nan')})


def test_read_provenance_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match='provenance.json'):
        read_provenance(tmp_path / 'nothing.csv')
    pd.DataFrame({'x': [1.0]}).to_parquet(tmp_path / 'plain.parquet')
    with pytest.raises(ValueError, match="'afx'"):
        read_provenance(tmp_path / 'plain.parquet')


def test_cli_prints_hash_and_provenance(audio_folder, tmp_path, capsys):
    out = str(tmp_path / 'f.parquet')
    assert cli.main([str(audio_folder), '--output', out, '--features', 'zcr']) == 0
    record = read_provenance(out)
    assert capsys.readouterr().out.strip().endswith(f"config hash {record['config_hash']}")

    assert cli.main([str(audio_folder), '--output', out, '--features', 'zcr', '--quiet', '--no-hash']) == 0
    assert capsys.readouterr().out == ''
    assert read_provenance(out)['file_sha256'] is None

    assert cli.main(['--provenance', out]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed['config_hash'] == record['config_hash']


def test_cli_provenance_error(tmp_path, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(['--provenance', os.fspath(tmp_path / 'missing.npy')])
    assert exc.value.code == 2
    assert 'No provenance record' in capsys.readouterr().err


def test_sklearn_provenance(audio_folder):
    pytest.importorskip('sklearn')
    import sklearn

    from afx_audio.sklearn import AFXFeatureExtractor

    est = AFXFeatureExtractor(config=CONFIG).fit([str(audio_folder / 'a.wav')])
    record = est.provenance_
    assert record['config_hash'] == config_hash(est.config_)
    assert record['config'] == json.loads(json.dumps(est.config_))
    assert record['sample_rate'] == SR
    assert record['versions']['scikit-learn'] == sklearn.__version__
    assert record['afx_version'] == afx_audio.__version__


def test_provenance_preset_id(audio_folder, tmp_path, capsys):
    out = str(tmp_path / 'p.json')
    extract_features_from_folder(str(audio_folder), afx_audio.load_preset('machinery@1'), out, verbose=False)
    record = read_provenance(out)
    assert (record['preset_id'], record['preset_modified']) == ('machinery@1', False)
    assert record['config_hash'] == config_hash(afx_audio.load_preset('machinery@1'))

    assert cli.main([str(audio_folder), '--preset', 'machinery@1', '--features', 'zcr', '-o', out]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-2:] == [
        'preset machinery@1 (modified)', f"config hash {read_provenance(out)['config_hash']}"]
    assert read_provenance(out)['preset_modified'] is True

    extract_features_from_folder(str(audio_folder), CONFIG, out, verbose=False)
    assert (read_provenance(out)['preset_id'], read_provenance(out)['preset_modified']) == (None, None)
    # A preset block that is not a bundled preset is reported but cannot be compared
    extract_features_from_folder(str(audio_folder), {**CONFIG, 'preset': {'name': 'lab', 'version': 3}}, out,
                                 verbose=False)
    assert (read_provenance(out)['preset_id'], read_provenance(out)['preset_modified']) == ('lab@3', None)


def test_sklearn_provenance_preset_id(audio_folder):
    pytest.importorskip('sklearn')
    from afx_audio.sklearn import AFXFeatureExtractor

    est = AFXFeatureExtractor(config='speech@1').fit([str(audio_folder / 'a.wav')])
    assert (est.provenance_['preset_id'], est.provenance_['preset_modified']) == ('speech@1', False)
    est = AFXFeatureExtractor(config='speech@1', aggregation='mean').fit([str(audio_folder / 'a.wav')])
    assert est.provenance_['preset_modified'] is True
    with pytest.raises(ValueError, match='Available'):
        AFXFeatureExtractor(config='speech@7').fit([str(audio_folder / 'a.wav')])
