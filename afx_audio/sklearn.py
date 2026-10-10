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
import json
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
from afx_audio.provenance import canonical_json, config_hash, environment_record, preset_record
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
            a config JSON, a preset ('speech' or 'speech@1', ...), or None for the
            bundled default config.
        features: Feature names to extract, in output order (names missing
            from the config get default parameters), or a dict of names to
            parameters that replaces the config's features block.
        sample_rate: Target sample rate for loading/assuming signals.
            None takes the config's sample_rate (22050 if it has none).
        aggregation: Temporal aggregation per file ('mean', 'std',
            'summary', 'advanced_summary', 'skew', 'kurtosis', 'stack').
            None takes the config's aggregation ('mean' if it has none).
        normalize: Per-sample normalization applied by AFX ('zscore' or
            'minmax'). None takes the config's normalize (none if it has
            none). This normalizes each sample's feature vector
            independently; for train/test consistent scaling, leave it
            unset in both and add a ``StandardScaler`` step to the
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
        provenance_: afx-audio, numpy, scipy and Python versions, the
            resolved config and its ``afx_audio.config_hash``, the preset
            id, the sample rate and the fit time, as written next to batch
            outputs.
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
        elif not os.path.isfile(self.config) and self.config.partition('@')[0] in list_presets():
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
        if self.normalize is not None:
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
        import sklearn

        environment = environment_record()
        environment['versions']['scikit-learn'] = sklearn.__version__
        self.provenance_ = {
            'afx_version': environment['versions']['afx_audio'],
            'config_hash': config_hash(self.config_),
            **preset_record(self.config_),
            'config': json.loads(canonical_json(self.config_)),
            'sample_rate': self.config_['sample_rate'],
            **environment,
        }
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


def scaler_from_stats(stats: dict[str, Any], columns: Sequence[str] | None = None):
    """
    A fitted ``StandardScaler`` from ``afx_audio.load_stats`` output.

    Equivalent to ``StandardScaler().fit(X)`` on the matrix the statistics
    came from (NaN ignored in fit, as sklearn does), without loading that
    matrix. ``columns`` selects and orders a subset of ``stats['columns']``;
    the scaler then expects data with those columns in that order. A
    column with no non-NaN value has no mean or scale, so it raises
    ValueError unless left out through ``columns``.
    """
    from sklearn.preprocessing import StandardScaler

    all_columns = list(stats['columns'])
    if columns is None:
        idx = np.arange(len(all_columns))
    else:
        position = {name: i for i, name in enumerate(all_columns)}
        missing = [c for c in columns if c not in position]
        if missing:
            raise ValueError(f'Columns not in the statistics: {missing[:5]}')
        idx = np.array([position[c] for c in columns], dtype=np.intp)
    count = np.asarray(stats['count'], dtype=np.int64)[idx]
    mean = np.asarray(stats['mean'], dtype=np.float64)[idx]
    var = np.asarray(stats['var'], dtype=np.float64)[idx]
    empty = count == 0
    if empty.any():
        names = [all_columns[i] for i in idx[empty]]
        raise ValueError(
            f'{len(names)} column(s) have no non-NaN value in the dataset, e.g. {names[:5]}; '
            'pass columns= without them and drop them from the data as well'
        )

    # Same rule as StandardScaler: a variance within rounding error of zero
    # gives scale 1 (sklearn.preprocessing._data._is_constant_feature)
    eps = np.finfo(np.float64).eps
    constant = var <= count * eps * var + (count * mean * eps) ** 2
    scale = np.sqrt(var)
    scale[constant | (scale < 10 * eps)] = 1.0

    scaler = StandardScaler()
    scaler.mean_ = mean
    scaler.var_ = var
    scaler.scale_ = scale
    scaler.n_features_in_ = len(idx)
    # sklearn stores a per-column count only when the data had missing values
    n_rows = int(stats['n_rows'])
    scaler.n_samples_seen_ = count if (count != n_rows).any() else n_rows
    return scaler
