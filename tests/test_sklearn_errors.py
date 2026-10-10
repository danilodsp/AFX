"""
AFXFeatureExtractor input errors, file configs and stacked output; PCA selector errors.
"""
import json

import numpy as np
import pytest

pytest.importorskip('sklearn')

from afx_audio.sklearn import AFXFeatureExtractor  # noqa: E402
from afx_audio.utils.selectors import pca_reducer  # noqa: E402

SR = 8000
CONFIG = {'sample_rate': SR, 'features': {'zcr': {'frame_size': 256, 'hop_length': 128}}}


def _signals(*seconds):
    rng = np.random.default_rng(0)
    return [rng.standard_normal(int(s * SR)) for s in seconds]


def test_fit_rejects_empty_input():
    with pytest.raises(ValueError, match='at least one sample'):
        AFXFeatureExtractor(config=CONFIG).fit([])


def test_fit_rejects_a_non_sequence():
    with pytest.raises(ValueError, match='sequence of paths or signals'):
        AFXFeatureExtractor(config=CONFIG).fit(5)


def test_names_before_fit():
    with pytest.raises(RuntimeError, match='Call fit before get_feature_names_out'):
        AFXFeatureExtractor(config=CONFIG).get_feature_names_out()


def test_config_from_a_file_matches_the_dict(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(CONFIG))
    X = _signals(0.5, 0.5)
    from_file = AFXFeatureExtractor(config=str(path)).fit_transform(X)
    np.testing.assert_array_equal(from_file, AFXFeatureExtractor(config=CONFIG).fit_transform(X))


def test_stack_keeps_every_frame_and_rejects_another_length():
    ext = AFXFeatureExtractor(config={**CONFIG, 'aggregation': 'stack'})
    out = ext.fit_transform(_signals(0.5, 0.5))
    assert out.shape == (2, 1 + (SR // 2 - 256) // 128)
    with pytest.raises(ValueError, match='does not match fitted length.*pooling aggregation'):
        ext.transform(_signals(1.0))


def test_pca_needs_at_least_one_key():
    with pytest.raises(ValueError, match='No feature keys'):
        pca_reducer({'a': np.zeros((2, 10))}, feature_keys=[])
