"""
scikit-learn compatible transformer for AFX feature extraction.

``AFXFeatureExtractor`` drops AFX straight into sklearn Pipelines,
GridSearchCV and cross-validation::

    from afx_audio.sklearn import AFXFeatureExtractor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    pipe = make_pipeline(
        AFXFeatureExtractor(features=['mfcc', 'spectral_centroid', 'zcr'],
                            sample_rate=16000, n_jobs=-1),
        StandardScaler(),
        SVC(),
    )
    pipe.fit(wav_paths_train, y_train)     # X can be file paths...
    pipe.predict(wav_paths_test)

    pipe.fit(signals, y)                   # ...or raw 1-D signals
"""
import copy
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

try:
    from sklearn.base import BaseEstimator, TransformerMixin
except ImportError as e:
    raise ImportError('afx_audio.sklearn requires scikit-learn: pip install "afx-audio[sklearn]"') from e

from afx_audio.extract_all import FEATURE_MAP, extract_all_features
from afx_audio.io.io import load_audio
from afx_audio.presets import list_presets, load_preset
from afx_audio.registry import custom_features, restore_custom_features
from afx_audio.utils.batch_utils import feature_names, features_to_vector
from afx_audio.utils.config_loader import load_config, unknown_feature_error, validate_config

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'config.json'
)


class AFXFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract AFX audio features as fixed-length vectors, sklearn-style.

    Each input sample is either a path to an audio file (str / os.PathLike,
    loaded and resampled to ``sample_rate``) or a raw 1-D signal
    (assumed to already be at ``sample_rate``). ``transform`` returns a
    2-D array of shape (n_samples, n_features_out).

    Parameters:
        config: Config dict (``extract_all_features`` convention), path to
            a config JSON, a preset name ('speech', ...), or None for the
            bundled default config.
        features: Feature names to extract, in output order (names missing
            from the config get default parameters), or a dict of names to
            parameters that replaces the config's features block.
        sample_rate: Target sample rate for loading/assuming signals.
            None takes the config's sample_rate (22050 if it has none).
        aggregation: Temporal aggregation per file ('mean', 'std',
            'summary', 'advanced_summary', 'skew', 'kurtosis', 'stack').
            None takes the config's aggregation ('mean' if it has none).
        normalize: Per-sample normalization applied by AFX ('zscore',
            'minmax', or None). Note this normalizes each sample's feature
            vector independently; for train/test consistent scaling,
            prefer ``normalize=None`` and a ``StandardScaler`` step in the
            pipeline.
        n_jobs: Parallel extraction over samples via joblib
            (None/1 = serial, -1 = all cores).
        coverage: Add a <feature>_valid_frac column after each feature's
            statistics, the fraction of frames with a finite value. None
            takes the config's setting (off if it has none).

    Attributes (after fit):
        config_: The resolved config actually used for extraction.
        feature_names_out_: Expanded output column names (vector features
            become ``name_0, name_1, ...``).
        n_features_out_: Output dimensionality.
    """

    def __init__(
        self,
        config: dict[str, Any] | str | None = None,
        features: list[str] | dict[str, dict[str, Any]] | None = None,
        sample_rate: int | None = None,
        aggregation: str | None = None,
        normalize: str | None = None,
        n_jobs: int | None = None,
        coverage: bool | None = None,
    ):
        self.config = config
        self.features = features
        self.sample_rate = sample_rate
        self.aggregation = aggregation
        self.normalize = normalize
        self.n_jobs = n_jobs
        self.coverage = coverage

    # -- config resolution --------------------------------------------------

    def _resolve_config(self) -> dict[str, Any]:
        if self.config is None:
            config = load_config(_DEFAULT_CONFIG)
        elif isinstance(self.config, dict):
            config = validate_config(copy.deepcopy(self.config), require_sample_rate=False)
        elif not os.path.isfile(self.config) and self.config in list_presets():
            config = load_preset(self.config)
        else:
            config = load_config(self.config)

        if isinstance(self.features, dict):
            config['features'] = copy.deepcopy(self.features)
        elif self.features is not None:
            for name in self.features:
                if name not in FEATURE_MAP:
                    raise unknown_feature_error(name)
            # Output columns follow the order of `features`; names missing
            # from the config get default parameters
            config['features'] = {f: config['features'].get(f, {}) for f in self.features}

        if self.sample_rate is not None:
            config['sample_rate'] = self.sample_rate
        config.setdefault('sample_rate', 22050)
        if self.aggregation is not None:
            config['aggregation'] = self.aggregation
        config.setdefault('aggregation', 'mean')
        config['preserve_shape'] = False
        config['flatten'] = False  # flattening handled by features_to_vector
        config['normalize'] = self.normalize
        if self.coverage is not None:
            config['coverage'] = self.coverage
        return validate_config(config)

    # -- per-sample extraction ----------------------------------------------

    def _load_signal(self, item, sr: int) -> np.ndarray:
        if isinstance(item, (str, os.PathLike)):
            signal, _ = load_audio(os.fspath(item), sr=sr)
            return signal
        # No cast or ravel: extract_all_features rejects multichannel and integer input
        return np.asarray(item)

    def _extract_dict(self, item, config) -> dict[str, np.ndarray]:
        signal = self._load_signal(item, config['sample_rate'])
        feats = extract_all_features(signal, config['sample_rate'], config)
        if isinstance(feats, np.ndarray):  # aggregation='stack'
            return {'stack': feats}
        return feats

    def _extract_vector(self, item, config, custom=None) -> np.ndarray:
        if custom:
            restore_custom_features(custom)
        return features_to_vector(self._extract_dict(item, config),
                                  flatten=True)

    # -- sklearn API --------------------------------------------------------

    def fit(self, X: Sequence, y=None):
        """Resolve the config and learn output feature names from X[0]."""
        if not hasattr(X, '__len__') and not hasattr(X, '__iter__'):
            raise ValueError('X must be a sequence of paths or signals')
        X = list(X)
        if len(X) == 0:
            raise ValueError('X must contain at least one sample')

        self.config_ = self._resolve_config()

        names = feature_names(self._extract_dict(X[0], self.config_))
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self.n_features_out_ = len(names)
        return self

    def transform(self, X: Sequence) -> np.ndarray:
        """Extract features for each sample; returns (n_samples, n_out)."""
        if not hasattr(self, 'config_'):
            raise RuntimeError(
                'This AFXFeatureExtractor instance is not fitted yet. '
                "Call 'fit' before 'transform'."
            )
        X = list(X)
        if self.n_jobs in (None, 1) or len(X) <= 1:
            vectors = [self._extract_vector(item, self.config_)
                       for item in X]
        else:
            from joblib import Parallel, delayed
            # joblib's workers are fresh processes without registered custom features
            custom = custom_features()
            vectors = Parallel(n_jobs=self.n_jobs)(
                delayed(self._extract_vector)(item, self.config_, custom)
                for item in X
            )

        out = np.stack(vectors, axis=0) if vectors else np.empty(
            (0, self.n_features_out_)
        )
        if out.shape[1] != self.n_features_out_:
            raise ValueError(
                f'Extracted vector length {out.shape[1]} does not match '
                f'fitted length {self.n_features_out_}. This can happen '
                f'when inputs have wildly different durations with '
                f"aggregation='stack'; use a pooling aggregation "
                f"('mean', 'summary', ...) for variable-length audio."
            )
        return out

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Expanded output column names (sklearn convention)."""
        if not hasattr(self, 'feature_names_out_'):
            raise RuntimeError('Call fit before get_feature_names_out.')
        return self.feature_names_out_

    # Inputs are file paths or ragged signals, not a numeric matrix.
    # _more_tags is read by sklearn < 1.6, __sklearn_tags__ by >= 1.6.
    def _more_tags(self):
        return {'X_types': ['string'], 'no_validation': True,
                'requires_fit': True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.requires_fit = True
        return tags
