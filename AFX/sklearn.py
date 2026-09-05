"""
scikit-learn compatible transformer for AFX feature extraction.

``AFXFeatureExtractor`` drops AFX straight into sklearn Pipelines,
GridSearchCV and cross-validation::

    from AFX.sklearn import AFXFeatureExtractor
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
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from AFX.extract_all import extract_all_features, FEATURE_MAP
from AFX.io.io import load_audio
from AFX.utils.batch_utils import features_to_vector
from AFX.utils.config_loader import load_config

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
            a config JSON, or None for the bundled default config.
        features: Optional list of feature names to extract (subset of the
            config's features block; unknown names raise at fit time).
        sample_rate: Target sample rate for loading/assuming signals.
        aggregation: Temporal aggregation per file ('mean', 'std',
            'summary', 'advanced_summary', 'skew', 'kurtosis', 'stack').
        normalize: Per-sample normalization applied by AFX ('zscore',
            'minmax', or None). Note this normalizes each sample's feature
            vector independently — for train/test consistent scaling,
            prefer ``normalize=None`` and a ``StandardScaler`` step in the
            pipeline.
        n_jobs: Parallel extraction over samples via joblib
            (None/1 = serial, -1 = all cores).

    Attributes (after fit):
        config_: The resolved config actually used for extraction.
        feature_names_out_: Expanded output column names (vector features
            become ``name_0, name_1, ...``).
        n_features_out_: Output dimensionality.
    """

    def __init__(
        self,
        config: Optional[Union[Dict[str, Any], str]] = None,
        features: Optional[List[str]] = None,
        sample_rate: int = 22050,
        aggregation: str = 'mean',
        normalize: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        self.config = config
        self.features = features
        self.sample_rate = sample_rate
        self.aggregation = aggregation
        self.normalize = normalize
        self.n_jobs = n_jobs

    # -- config resolution --------------------------------------------------

    def _resolve_config(self) -> Dict[str, Any]:
        if self.config is None:
            config = load_config(_DEFAULT_CONFIG)
        elif isinstance(self.config, dict):
            config = copy.deepcopy(self.config)
        else:
            config = load_config(self.config)

        if self.features is not None:
            known = set(config['features'])
            unknown = [f for f in self.features if f not in known]
            # Allow features that exist in FEATURE_MAP but not in the
            # config: add them with default parameters.
            addable = [f for f in unknown if f in FEATURE_MAP]
            really_unknown = [f for f in unknown if f not in FEATURE_MAP]
            if really_unknown:
                raise ValueError(
                    f'Unknown feature(s): {really_unknown}. '
                    f'Available: {sorted(FEATURE_MAP)}'
                )
            config['features'] = {
                **{k: v for k, v in config['features'].items()
                   if k in self.features},
                **{f: {} for f in addable},
            }

        config['sample_rate'] = self.sample_rate
        config['preserve_shape'] = False
        config['aggregation'] = self.aggregation
        config['flatten'] = False  # flattening handled by features_to_vector
        config['normalize'] = self.normalize
        return config

    # -- per-sample extraction ----------------------------------------------

    def _load_signal(self, item) -> np.ndarray:
        if isinstance(item, (str, os.PathLike)):
            signal, _ = load_audio(os.fspath(item), sr=self.sample_rate)
            return signal
        signal = np.asarray(item, dtype=np.float64).ravel()
        return signal

    def _extract_dict(self, item, config) -> Dict[str, np.ndarray]:
        signal = self._load_signal(item)
        feats = extract_all_features(signal, self.sample_rate, config)
        if isinstance(feats, np.ndarray):  # aggregation='stack'
            return {'stack': feats}
        return feats

    def _extract_vector(self, item, config) -> np.ndarray:
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

        probe = self._extract_dict(X[0], self.config_)
        names: List[str] = []
        for name, values in probe.items():
            size = int(np.asarray(values).ravel().shape[0])
            if size == 1:
                names.append(name)
            else:
                names.extend(f'{name}_{i}' for i in range(size))
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
            vectors = Parallel(n_jobs=self.n_jobs)(
                delayed(self._extract_vector)(item, self.config_)
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

    def _more_tags(self):
        # Inputs are file paths or ragged signals, not a numeric matrix.
        return {'X_types': ['string'], 'no_validation': True,
                'requires_fit': True}
