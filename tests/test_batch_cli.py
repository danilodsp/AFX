"""
Tests for batch extraction (extract_features_from_folder) and the CLI.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from afx_audio import cli
from afx_audio.batch_extractor import extract_features_from_folder

SR = 22050


@pytest.fixture
def audio_folder(tmp_path):
    """Folder with two same-named wavs in different subfolders + one at root."""
    rng = np.random.RandomState(0)
    t = np.linspace(0, 0.5, SR // 2, endpoint=False)
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    noise = (0.1 * rng.randn(SR // 2)).astype(np.float32)

    (tmp_path / 'a').mkdir()
    (tmp_path / 'b').mkdir()
    sf.write(tmp_path / 'root.wav', tone, SR)
    sf.write(tmp_path / 'a' / 'take1.wav', tone, SR)
    sf.write(tmp_path / 'b' / 'take1.wav', noise, SR)
    return tmp_path


@pytest.fixture
def small_config(tmp_path):
    config = {
        'sample_rate': SR,
        'features': {
            'zcr': {'frame_size': 2048, 'hop_length': 512},
            'rms_energy': {'frame_size': 2048, 'hop_length': 512},
            'mfcc': {'n_mfcc': 5, 'frame_size': 2048, 'hop_length': 512},
        },
        'aggregation': 'mean',
    }
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return str(path)


def test_batch_npy_roundtrip_and_relpath_keys(audio_folder, small_config, tmp_path):
    out = str(tmp_path / 'features.npy')
    results = extract_features_from_folder(
        str(audio_folder), small_config, out, save_format='npy'
    )
    # Same-named files in different subfolders must NOT collide
    keys = set(results)
    assert os.path.join('a', 'take1.wav') in keys
    assert os.path.join('b', 'take1.wav') in keys
    assert 'root.wav' in keys
    assert len(keys) == 3
    # Documented load procedure works
    loaded = np.load(out, allow_pickle=True).item()
    assert set(loaded) == keys
    assert 'mfcc' in loaded['root.wav']


def test_batch_csv_scalar_columns(audio_folder, small_config, tmp_path):
    out = str(tmp_path / 'features.csv')
    extract_features_from_folder(
        str(audio_folder), small_config, out, save_format='csv'
    )
    df = pd.read_csv(out, index_col='file')
    assert len(df) == 3
    # Vector features expand to name_i scalar columns
    assert 'mfcc_0' in df.columns and 'mfcc_4' in df.columns
    # Scalar (aggregated 1-value) features keep their name
    assert 'zcr' in df.columns and 'rms_energy' in df.columns
    # All cells numeric (no stringified lists)
    assert df.select_dtypes(include=[np.number]).shape == df.shape


def test_batch_json_output(audio_folder, small_config, tmp_path):
    out = str(tmp_path / 'features.json')
    extract_features_from_folder(
        str(audio_folder), small_config, out, save_format='json'
    )
    with open(out) as f:
        data = json.load(f)
    assert len(data) == 3


def test_batch_unknown_feature_raises(audio_folder, small_config, tmp_path):
    with pytest.raises(ValueError, match='Unknown feature'):
        extract_features_from_folder(
            str(audio_folder), small_config, str(tmp_path / 'o.npy'),
            features=['mfcc', 'not_a_feature']
        )


def test_batch_unknown_aggregation_raises(audio_folder, small_config, tmp_path):
    with pytest.raises(ValueError, match='Unknown aggregation'):
        extract_features_from_folder(
            str(audio_folder), small_config, str(tmp_path / 'o.npy'),
            aggregation='bogus'
        )


def test_batch_stack_aggregation(audio_folder, small_config, tmp_path):
    out = str(tmp_path / 'features.json')
    results = extract_features_from_folder(
        str(audio_folder), small_config, out, save_format='json',
        aggregation='stack'
    )
    vec = results['root.wav']['stack']
    assert isinstance(vec, list) and len(vec) > 0


def test_batch_errors_separated(audio_folder, small_config, tmp_path):
    # Add a corrupt "wav"
    bad = audio_folder / 'broken.wav'
    bad.write_bytes(b'this is not audio')
    out = str(tmp_path / 'features.csv')
    with pytest.warns(UserWarning, match='1 file'):
        results = extract_features_from_folder(
            str(audio_folder), small_config, out, save_format='csv'
        )
    assert 'errors' in results and 'broken.wav' in results['errors']
    # Error rows are NOT mixed into the CSV
    df = pd.read_csv(out, index_col='file')
    assert 'broken.wav' not in df.index
    assert 'error' not in df.columns
    # And are written to the companion errors file
    with open(out + '.errors.json') as f:
        errors = json.load(f)
    assert 'broken.wav' in errors


def test_cli_main(audio_folder, small_config, tmp_path, monkeypatch):
    out = str(tmp_path / 'cli_out.json')
    monkeypatch.setattr(sys, 'argv', [
        'afx', str(audio_folder),
        '--config', small_config,
        '--output', out,
        '--format', 'json',
        '--features', ' mfcc, zcr ',  # whitespace must be stripped
    ])
    cli.main()
    with open(out) as f:
        data = json.load(f)
    assert set(data['root.wav']) == {'mfcc', 'zcr'}


def test_cli_default_config_is_package_relative():
    assert os.path.isabs(cli.DEFAULT_CONFIG)
    assert os.path.exists(cli.DEFAULT_CONFIG)


def test_batch_accepts_config_dict(audio_folder, tmp_path):
    from afx_audio.presets import load_preset
    config = load_preset('speech')
    out = str(tmp_path / 'preset.csv')
    extract_features_from_folder(
        str(audio_folder), config, out, save_format='csv',
        features=['mfcc', 'zcr'], verbose=False,
    )
    df = pd.read_csv(out, index_col='file')
    assert len(df) == 3
    assert 'mfcc_mean_0' in df.columns and 'zcr_mean' in df.columns
    # The caller's dict is left untouched
    assert set(config['features']) > {'mfcc', 'zcr'}


def test_cli_preset_and_format_inference(audio_folder, tmp_path, monkeypatch):
    out = str(tmp_path / 'out.csv')
    cli.main([str(audio_folder), '--preset', 'machinery',
              '--features', 'rms_energy,kurtosis', '--output', out, '-q'])
    df = pd.read_csv(out, index_col='file')
    assert len(df) == 3
    assert {'rms_energy_mean', 'kurtosis_kurt'} <= set(df.columns)


def test_cli_list_options(capsys):
    assert cli.main(['--list-features']) == 0
    assert 'mfcc' in capsys.readouterr().out.split()
    assert cli.main(['--list-presets']) == 0
    assert 'speech@1' in capsys.readouterr().out.split()


def test_cli_rejects_unknown_preset_version(audio_folder, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main([str(audio_folder), '--preset', 'speech@9'])
    assert exc.value.code == 2
    assert "Available: ['speech@1']" in capsys.readouterr().err


def test_cli_rejects_missing_folder(tmp_path):
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / 'nope')])


def test_cli_describe(capsys):
    assert cli.main(['--describe', 'spectral_rolloff']) == 0
    out = capsys.readouterr().out
    assert 'roll_percent = 0.85' in out and 'Hz' in out
    assert 'Torch: afx_audio.torch.functional.spectral_rolloff, equal to NumPy within rtol=1e-09' in out
    assert cli.main(['--describe', 'hnr']) == 0
    assert 'Torch: none' in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cli.main(['--describe', 'nope'])


def test_cli_version(capsys):
    import afx_audio

    with pytest.raises(SystemExit) as exc:
        cli.main(['--version'])
    assert exc.value.code == 0
    assert afx_audio.__version__ in capsys.readouterr().out


def test_cli_errors_are_one_line_with_status_2(audio_folder, tmp_path, capsys):
    cases = [
        (['--features', 'mfccs'], "Did you mean 'mfcc'"),
        (['--output', str(tmp_path / 'missing' / 'x.csv')], 'Output folder does not exist'),
        (['--output', str(tmp_path / 'x.xlsx')], 'Cannot infer the output format'),
        (['--config', str(tmp_path / 'nope.json')], 'Config file not found'),
    ]
    for extra, message in cases:
        with pytest.raises(SystemExit) as exc:
            cli.main([str(audio_folder), '-q', *extra])
        assert exc.value.code == 2
        err = capsys.readouterr().err
        assert err.startswith('afx: error:') and message in err and 'Traceback' not in err


def test_cli_status_1_when_nothing_extracted(tmp_path, capsys):
    empty = tmp_path / 'empty'
    empty.mkdir()
    assert cli.main([str(empty), '-q', '-o', str(tmp_path / 'e.csv')]) == 1
    assert 'no features were extracted' in capsys.readouterr().err


def test_cli_adds_features_missing_from_the_preset(audio_folder, tmp_path):
    out = tmp_path / 'g.csv'
    assert cli.main([str(audio_folder), '--preset', 'speech', '--features', 'gfcc', '-q', '-o', str(out)]) == 0
    assert pd.read_csv(out, index_col='file').columns[0] == 'gfcc_mean_0'


def test_json_output_writes_null_for_nan(audio_folder, tmp_path):
    import json

    silent = audio_folder / 'silent.wav'
    sf.write(silent, np.zeros(8000, dtype=np.float32), 16000)
    out = tmp_path / 'f.json'
    extract_features_from_folder(str(audio_folder), {'sample_rate': 16000, 'features': ['hnr']}, str(out),
                                 verbose=False)
    text = out.read_text()
    assert 'NaN' not in text
    assert json.loads(text)['silent.wav']['hnr'] == [None]


def test_api_infers_format_and_keeps_the_exact_path(audio_folder, tmp_path):
    out = tmp_path / 'features.data'
    extract_features_from_folder(str(audio_folder), {'sample_rate': 16000, 'features': ['zcr']}, str(out),
                                 save_format='npy', verbose=False)
    assert out.exists() and not (tmp_path / 'features.data.npy').exists()
    out_csv = tmp_path / 'f.csv'
    extract_features_from_folder(str(audio_folder), {'sample_rate': 16000, 'features': ['zcr']}, str(out_csv),
                                 verbose=False)
    assert out_csv.read_text().startswith('file,zcr')


def test_rerun_removes_sidecars_it_does_not_write(audio_folder, small_config, tmp_path):
    out = tmp_path / 'f.csv'
    (audio_folder / 'broken.wav').write_bytes(b'not audio')
    with pytest.warns(UserWarning, match='1 file'):
        extract_features_from_folder(str(audio_folder), small_config, str(out), dataset_stats=True, verbose=False)
    assert (tmp_path / 'f.csv.errors.json').exists() and (tmp_path / 'f.csv.stats.json').exists()

    (audio_folder / 'broken.wav').unlink()
    extract_features_from_folder(str(audio_folder), small_config, str(out), verbose=False)
    assert not (tmp_path / 'f.csv.errors.json').exists()
    assert not (tmp_path / 'f.csv.stats.json').exists()
    assert (tmp_path / 'f.csv.provenance.json').exists()

    # Parquet carries its record inside, so a sidecar from an earlier csv run under the same name goes
    pytest.importorskip('pyarrow')
    extract_features_from_folder(str(audio_folder), small_config, str(out), save_format='parquet', verbose=False)
    assert not (tmp_path / 'f.csv.provenance.json').exists()
