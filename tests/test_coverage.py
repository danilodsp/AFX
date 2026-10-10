"""
Tests for per-frame validity (NaN on undefined frames) and the clip-level coverage option.
"""
import json

import numpy as np
import pytest
import soundfile as sf

import afx_audio
from afx_audio import cli
from afx_audio.batch_extractor import extract_features_from_folder
from afx_audio.utils.aggregator import aggregate_features, valid_fraction, valid_frame_mask

SR = 16000
FRAMES = {'frame_size': 1024, 'hop_length': 256}


def _half_silent(seconds=1.0, f0=220.0):
    n = int(seconds * SR)
    tone = 0.5 * np.sin(2 * np.pi * f0 * np.arange(n) / SR)
    return np.concatenate([np.zeros(n), tone])


def test_valid_frame_mask_and_fraction():
    assert valid_fraction(np.array([1.0, np.nan, 2.0, np.inf])) == 0.5
    # A frame counts only when every row is finite
    rows = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, np.nan, 3.0, 4.0]])
    np.testing.assert_array_equal(valid_frame_mask(rows), [True, False, True, True])
    assert valid_fraction(rows) == 0.75
    assert valid_fraction(np.float64(np.nan)) == 0.0
    assert valid_fraction(np.float64(1.0)) == 1.0
    assert afx_audio.valid_fraction is valid_fraction


@pytest.mark.parametrize('name', ['pitch', 'hnr'])
def test_valid_frac_half_silence_half_tone(name):
    x = _half_silent()
    out = afx_audio.extract_all_features(x, SR, {'features': {name: FRAMES}, 'coverage': True})
    frames = afx_audio.FEATURE_MAP[name](x, SR, **FRAMES)[name]
    assert out[f'{name}_valid_frac'][0] == pytest.approx(valid_fraction(frames))
    assert out[f'{name}_valid_frac'][0] == pytest.approx(0.5, abs=0.03)
    # The statistic is over the tone only
    if name == 'pitch':
        assert out['pitch'][0] == pytest.approx(220.0, rel=0.01)


def test_coverage_off_by_default_and_identical():
    x = _half_silent()
    features = {'pitch': FRAMES, 'hnr': FRAMES, 'mfcc': {'n_mfcc': 5}}
    base = afx_audio.extract_all_features(x, SR, {'features': features, 'aggregation': 'summary'})
    with_cov = afx_audio.extract_all_features(x, SR, {'features': features, 'aggregation': 'summary',
                                                      'coverage': True})
    assert not any(k.endswith('_valid_frac') for k in base)
    assert [k for k in with_cov if not k.endswith('_valid_frac')] == list(base)
    for k, v in base.items():
        np.testing.assert_array_equal(with_cov[k], v)


@pytest.mark.parametrize('aggregation, stats', [
    ('mean', ['']),
    ('std', ['']),
    ('skew', ['']),
    ('kurtosis', ['']),
    ('summary', ['_mean', '_std']),
    ('advanced_summary', ['_mean', '_std', '_skew', '_kurt']),
])
def test_valid_frac_follows_each_feature(aggregation, stats):
    features = {'hnr': FRAMES, 'mfcc': {'n_mfcc': 4}, 'mfcc_delta': {'n_mfcc': 4}}
    out = afx_audio.extract_all_features(_half_silent(), SR,
                                         {'features': features, 'aggregation': aggregation, 'coverage': True})
    expected = [k for name in features for k in [name + s for s in stats] + [f'{name}_valid_frac']]
    assert list(out) == expected
    assert out['mfcc_valid_frac'][0] == 1.0


def test_valid_frac_for_multi_row_feature_uses_all_rows():
    x = _half_silent()
    out = afx_audio.extract_all_features(x, SR, {'features': {'formants': {}}, 'coverage': True})
    formants = afx_audio.FEATURE_MAP['formants'](x, SR)['formants']
    assert out['formants_valid_frac'][0] == pytest.approx(np.isfinite(formants).all(axis=0).mean())
    assert out['formants_valid_frac'][0] <= np.isfinite(formants[0]).mean()


def test_valid_frac_is_not_normalized():
    features = {'a': np.array([1.0, np.nan, 3.0, 5.0]), 'b': np.array([[1.0, 2.0], [3.0, np.nan]])}
    stats = {'mean': {'a': 10.0, 'a_valid_frac': 10.0}, 'std': {'a': 2.0, 'a_valid_frac': 2.0}}
    out = aggregate_features(features, method='mean', normalize='zscore', norm_stats=stats, coverage=True)
    assert out['a_valid_frac'][0] == 0.75
    assert out['b_valid_frac'][0] == 0.5
    assert out['a'][0] == pytest.approx((3.0 - 10.0) / 2.0)


def test_valid_frac_per_window():
    v = np.array([np.nan, np.nan, 1.0, 2.0, 3.0, np.nan])
    out = aggregate_features({'x': v}, method='sliding_window', window_size=2, hop_length=2, coverage=True)
    np.testing.assert_array_equal(out['x_valid_frac'], [0.0, 1.0, 0.5])
    assert out['x'].shape == out['x_valid_frac'].shape


def test_valid_frac_unflattened_shape():
    features = {'a': np.ones((3, 5)), 'b': np.ones(5)}
    out = aggregate_features(features, method='summary', flatten=False, coverage=True)
    assert out['a_valid_frac'].shape == (1,) and out['b_valid_frac'].shape == (1,)


@pytest.mark.parametrize('config, match', [
    ({'coverage': 'yes'}, 'true or false'),
    ({'coverage': 1}, 'true or false'),
    ({'coverage': True, 'aggregation': 'stack'}, 'stack'),
    ({'coverage': True, 'preserve_shape': True}, 'preserve_shape'),
])
def test_coverage_config_validation(config, match):
    with pytest.raises(ValueError, match=match):
        afx_audio.validate_config({'features': ['zcr'], **config}, require_sample_rate=False)


def test_aggregate_features_rejects_coverage_with_stack():
    with pytest.raises(ValueError, match='stack'):
        aggregate_features({'a': np.ones(3)}, method='stack', coverage=True)


def test_flatten_features_names():
    out = afx_audio.extract_all_features(_half_silent(), SR, {
        'features': {'hnr': FRAMES, 'mfcc': {'n_mfcc': 3}}, 'aggregation': 'summary', 'coverage': True})
    names, values = afx_audio.flatten_features(out)
    assert names == afx_audio.feature_names(out)
    assert names == ['hnr_mean', 'hnr_std', 'hnr_valid_frac', 'mfcc_mean_0', 'mfcc_mean_1', 'mfcc_mean_2',
                     'mfcc_std_0', 'mfcc_std_1', 'mfcc_std_2', 'mfcc_valid_frac']
    assert values.shape == (len(names),)


def test_extract_segments_reports_coverage_per_segment():
    x = _half_silent()
    names, X, starts = afx_audio.extract_segments(x, SR, {'features': {'pitch': FRAMES}, 'coverage': True},
                                                  segment_length=0.5)
    assert names == ['pitch', 'pitch_valid_frac']
    np.testing.assert_array_equal(X[:2, 1], 0.0)
    assert np.isnan(X[:2, 0]).all()
    assert (X[2:, 1] > 0.95).all()
    np.testing.assert_allclose(starts, [0.0, 0.5, 1.0, 1.5])


def test_pitch_unvoiced_frames_are_nan_with_voiced_mask():
    x = _half_silent()
    out = afx_audio.FEATURE_MAP['pitch'](x, SR, return_metadata=True, **FRAMES)
    voiced = out['metadata']['voiced']
    assert voiced.dtype == bool and voiced.shape == out['pitch'].shape
    np.testing.assert_array_equal(voiced, np.isfinite(out['pitch']))
    assert not voiced[:40].any() and voiced[-40:].all()
    # A tone below the search range is unvoiced too
    assert np.isnan(afx_audio.FEATURE_MAP['pitch'](x, SR, fmin=600.0, **FRAMES)['pitch']).all()


def test_metadata_reaches_extract_all_features():
    out = afx_audio.extract_all_features(_half_silent(), SR, {'features': {'pitch': FRAMES}}, return_metadata=True)
    assert out['metadata']['pitch']['voiced'].sum() > 0


def test_every_nan_producer_documents_when():
    rng = np.random.default_rng(0)
    signals = [np.zeros(SR), _half_silent(0.5), 0.1 * rng.standard_normal(SR)]
    for name in afx_audio.list_features():
        info = afx_audio.feature_info(name)
        produces_nan = any(
            not np.isfinite(afx_audio.FEATURE_MAP[name](s, SR)[name]).all() for s in signals
        )
        if produces_nan:
            assert info['undefined'], name


def test_batch_extractor_writes_valid_frac_columns(tmp_path):
    folder = tmp_path / 'audio'
    folder.mkdir()
    sf.write(folder / 'half.wav', _half_silent().astype(np.float32), SR, subtype='FLOAT')
    sf.write(folder / 'silent.wav', np.zeros(SR, dtype=np.float32), SR, subtype='FLOAT')
    config = {'sample_rate': SR, 'features': {'hnr': FRAMES, 'zcr': {}}, 'aggregation': 'summary'}
    out = tmp_path / 'f.csv'
    extract_features_from_folder(str(folder), config, str(out), coverage=True, verbose=False)
    header, *rows = out.read_text().splitlines()
    assert header.split(',') == ['file', 'hnr_mean', 'hnr_std', 'hnr_valid_frac', 'zcr_mean', 'zcr_std',
                                 'zcr_valid_frac']
    cells = {r.split(',')[0]: r.split(',') for r in rows}
    assert float(cells['half.wav'][3]) == pytest.approx(0.5, abs=0.03)
    assert float(cells['silent.wav'][3]) == 0.0

    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))
    out_cli = tmp_path / 'cli.csv'
    assert cli.main([str(folder), '--config', str(config_path), '-o', str(out_cli), '--coverage', '-q']) == 0
    assert out_cli.read_text() == out.read_text()


def test_sklearn_feature_names_include_valid_frac():
    pytest.importorskip('sklearn')
    from afx_audio.sklearn import AFXFeatureExtractor

    signals = [_half_silent(0.5), _half_silent(0.5, f0=330.0)]
    est = AFXFeatureExtractor(features={'pitch': FRAMES, 'zcr': {}}, sample_rate=SR, aggregation='summary',
                              coverage=True)
    X = est.fit_transform(signals)
    names = list(est.get_feature_names_out())
    assert names == ['pitch_mean', 'pitch_std', 'pitch_valid_frac', 'zcr_mean', 'zcr_std', 'zcr_valid_frac']
    assert X.shape == (2, 6)
    np.testing.assert_allclose(X[:, 2], 0.5, atol=0.05)
    assert 'coverage' in est.get_params()


def test_torch_module_valid_frac_matches_numpy():
    torch = pytest.importorskip('torch')
    from afx_audio.torch import AFXFeatureModule

    # Noise, not a tone: the spectral kurtosis of a pure tone is too ill-conditioned to compare
    x = np.concatenate([np.zeros(SR), 0.1 * np.random.default_rng(1).standard_normal(SR)])
    features = {'spectral_kurtosis': {}, 'mfcc': {'n_mfcc': 4}}
    module = AFXFeatureModule(features, sample_rate=SR, aggregation='mean_std', output='vector',
                              coverage=True).double()
    out = module(torch.as_tensor(x)).numpy()
    ref = afx_audio.extract_all_features(x, SR, {'features': features, 'aggregation': 'summary', 'coverage': True})
    names, values = afx_audio.flatten_features(ref)
    assert module.get_feature_names_out() == names
    np.testing.assert_allclose(out, values, rtol=1e-6, atol=1e-6)
    assert ref['spectral_kurtosis_valid_frac'][0] == pytest.approx(0.5, abs=0.03)

    batch = module(torch.as_tensor(np.stack([x, x])))
    assert batch.shape == (2, len(names))
    feats = AFXFeatureModule(features, sample_rate=SR, aggregation='mean', coverage=True).double()(
        torch.as_tensor(np.stack([x, x])))
    assert feats['mfcc_valid_frac'].shape == (2,)
    with pytest.raises(ValueError, match='coverage'):
        AFXFeatureModule(features, sample_rate=SR, coverage=True)
