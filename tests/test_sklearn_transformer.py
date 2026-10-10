"""
Tests for afx_audio.sklearn.AFXFeatureExtractor: fit/transform semantics,
pipeline integration, feature names, and parallel extraction.
"""

import numpy as np
import pytest
import soundfile as sf

from afx_audio.sklearn import AFXFeatureExtractor

SR = 16000
RNG = np.random.default_rng(0)

FEATURES = ['mfcc', 'spectral_centroid', 'zcr', 'rms_energy']
CONFIG = {
    'sample_rate': SR,
    'features': {
        'mfcc': {'n_mfcc': 13, 'frame_size': 1024, 'hop_length': 512},
        'spectral_centroid': {'frame_size': 1024, 'hop_length': 512},
        'zcr': {'frame_size': 1024, 'hop_length': 512},
        'rms_energy': {},
    },
}


def _make_signals(n=6, seconds=0.5):
    sigs = []
    for i in range(n):
        t = np.arange(int(SR * seconds)) / SR
        f0 = 200 + 150 * i
        sigs.append(
            np.sin(2 * np.pi * f0 * t) + 0.05 * RNG.standard_normal(t.size)
        )
    return sigs


@pytest.fixture
def signals():
    return _make_signals()


@pytest.fixture
def wav_paths(tmp_path, signals):
    paths = []
    for i, sig in enumerate(signals):
        p = tmp_path / f'tone_{i}.wav'
        sf.write(p, sig.astype(np.float32), SR, subtype='FLOAT')
        paths.append(str(p))
    return paths


def test_fit_transform_signals(signals):
    ext = AFXFeatureExtractor(config=CONFIG, sample_rate=SR)
    X = ext.fit_transform(signals)
    assert X.shape == (len(signals), ext.n_features_out_)
    # 13 mfcc + 1 centroid + 1 zcr + 1 rms with mean aggregation
    assert ext.n_features_out_ == 16
    assert np.isfinite(X).all()


def test_transform_on_file_paths(wav_paths, signals):
    ext = AFXFeatureExtractor(config=CONFIG, sample_rate=SR)
    X_paths = ext.fit_transform(wav_paths)
    X_sigs = ext.fit_transform(signals)
    assert X_paths.shape == X_sigs.shape
    # float32 wav round-trip: close but not identical
    np.testing.assert_allclose(X_paths, X_sigs, rtol=1e-2, atol=1e-2)


def test_feature_names_out(signals):
    ext = AFXFeatureExtractor(config=CONFIG, sample_rate=SR).fit(signals)
    names = list(ext.get_feature_names_out())
    assert len(names) == ext.n_features_out_
    assert 'mfcc_0' in names and 'mfcc_12' in names
    assert 'spectral_centroid' in names
    assert len(set(names)) == len(names)  # unique


def test_features_subset_param(signals):
    ext = AFXFeatureExtractor(config=CONFIG, features=['zcr', 'rms_energy'],
                              sample_rate=SR)
    X = ext.fit_transform(signals)
    assert X.shape[1] == 2


def test_feature_addable_from_feature_map(signals):
    # 'kurtosis' isn't in CONFIG but exists in FEATURE_MAP -> added with
    # default params
    ext = AFXFeatureExtractor(config=CONFIG,
                              features=['zcr', 'kurtosis'], sample_rate=SR)
    X = ext.fit_transform(signals)
    assert X.shape[1] == 2


def test_unknown_feature_raises(signals):
    ext = AFXFeatureExtractor(config=CONFIG, features=['not_a_feature'])
    with pytest.raises(ValueError, match='Unknown feature'):
        ext.fit(signals)


def test_unfitted_transform_raises(signals):
    with pytest.raises(RuntimeError, match='not fitted'):
        AFXFeatureExtractor(config=CONFIG).transform(signals)


def test_pipeline_end_to_end(signals):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = np.array([0, 0, 0, 1, 1, 1])
    pipe = make_pipeline(
        AFXFeatureExtractor(config=CONFIG, sample_rate=SR),
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    )
    pipe.fit(signals, y)
    preds = pipe.predict(signals)
    assert preds.shape == (6,)
    # Low/high frequency tones are trivially separable
    assert (preds == y).mean() == 1.0


def test_clone_and_get_params(signals):
    from sklearn.base import clone

    ext = AFXFeatureExtractor(config=CONFIG, features=['zcr'],
                              sample_rate=SR, aggregation='std')
    cloned = clone(ext)
    assert cloned.get_params()['aggregation'] == 'std'
    X1 = ext.fit_transform(signals)
    X2 = cloned.fit_transform(signals)
    np.testing.assert_allclose(X1, X2)


def test_n_jobs_matches_serial(signals):
    serial = AFXFeatureExtractor(config=CONFIG, sample_rate=SR)
    parallel = AFXFeatureExtractor(config=CONFIG, sample_rate=SR, n_jobs=2)
    X1 = serial.fit_transform(signals)
    X2 = parallel.fit_transform(signals)
    np.testing.assert_allclose(X1, X2, rtol=1e-12)


def test_summary_aggregation(signals):
    ext = AFXFeatureExtractor(config=CONFIG, features=['zcr'],
                              sample_rate=SR, aggregation='summary')
    X = ext.fit_transform(signals)
    names = list(ext.get_feature_names_out())
    assert X.shape[1] == 2  # zcr_mean, zcr_std
    assert names == ['zcr_mean', 'zcr_std']


def test_default_config_loads(signals):
    """config=None uses the bundled default config."""
    ext = AFXFeatureExtractor(features=['zcr', 'rms_energy'],
                              sample_rate=SR)
    X = ext.fit_transform(signals[:2])
    assert X.shape[0] == 2


def test_output_columns_follow_requested_feature_order():
    from afx_audio.sklearn import AFXFeatureExtractor

    x = [np.random.default_rng(0).standard_normal(8000)]
    ext = AFXFeatureExtractor(features=['mfcc', 'zcr'], sample_rate=16000, aggregation='summary').fit(x)
    names = list(ext.get_feature_names_out())
    assert names[0] == 'mfcc_mean_0' and names[-1] == 'zcr_std'


@pytest.mark.parametrize('bad', [np.zeros((8000, 2)), np.zeros(8000, dtype=np.int16)])
def test_raw_signals_are_validated(bad):
    from afx_audio.sklearn import AFXFeatureExtractor

    with pytest.raises((ValueError, TypeError)):
        AFXFeatureExtractor(features=['zcr'], sample_rate=16000).fit([bad])


def test_bad_aggregation_fails_before_extraction():
    from afx_audio.sklearn import AFXFeatureExtractor

    with pytest.raises(ValueError, match='aggregation'):
        AFXFeatureExtractor(features=['zcr'], aggregation='bogus').fit([np.zeros(100)])


def test_config_sample_rate_and_aggregation_are_used_by_default():
    import afx_audio

    preset = afx_audio.load_preset('speech')
    ext = AFXFeatureExtractor(config=preset).fit([np.zeros(SR)])
    assert ext.config_['sample_rate'] == preset['sample_rate'] == SR
    assert ext.config_['aggregation'] == 'summary'
    assert AFXFeatureExtractor(config=preset, aggregation='mean').fit([np.zeros(SR)]).config_['aggregation'] == 'mean'
    assert AFXFeatureExtractor(features=['zcr']).fit([np.zeros(100)]).config_['sample_rate'] == 22050


def test_pandas_output():
    pd = pytest.importorskip('pandas')
    x = [np.random.default_rng(i).standard_normal(4000) for i in range(3)]
    out = AFXFeatureExtractor(features=['zcr', 'mfcc'], sample_rate=SR).set_output(transform='pandas').fit_transform(x)
    assert isinstance(out, pd.DataFrame) and list(out.columns[:2]) == ['zcr', 'mfcc_0']


def test_features_dict_and_preset_name():
    x = [np.random.default_rng(0).standard_normal(SR)]
    ext = AFXFeatureExtractor(features={'mfcc': {'n_mfcc': 3}}, sample_rate=SR).fit(x)
    assert list(ext.get_feature_names_out()) == ['mfcc_0', 'mfcc_1', 'mfcc_2']
    with pytest.raises(ValueError, match="Did you mean 'n_mfcc'"):
        AFXFeatureExtractor(features={'mfcc': {'n_mfccs': 3}}).fit(x)
    ext = AFXFeatureExtractor(config='speech').fit(x)
    assert ext.config_['sample_rate'] == 16000 and ext.config_['aggregation'] == 'summary'


def test_config_normalize_is_kept_unless_overridden(signals):
    import afx_audio

    config = {**CONFIG, 'normalize': 'zscore'}
    est = AFXFeatureExtractor(config=config).fit(signals[:1])
    assert est.config_['normalize'] == 'zscore'
    reference = afx_audio.flatten_features(afx_audio.extract_all_features(signals[0], SR, config))[1]
    np.testing.assert_allclose(est.transform(signals[:1])[0], reference)
    assert est.provenance_['config_hash'] == afx_audio.config_hash(config)
    assert AFXFeatureExtractor(config=config, normalize='minmax').fit(signals[:1]).config_['normalize'] == 'minmax'
