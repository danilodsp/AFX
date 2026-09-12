"""
AFX: audio feature extraction for machine learning.

Top-level entry points: ``load_audio`` -> ``extract_all_features`` for one
signal, ``extract_features_from_folder`` (or the ``afx`` CLI) for a dataset,
``AFXFeatureExtractor`` for scikit-learn pipelines and ``AFX.torch`` for the
differentiable path. ``FEATURE_MAP`` lists every feature name.
"""
from importlib.metadata import PackageNotFoundError, version

from .extract_all import extract_all_features, FEATURE_MAP
from .extractors.cepstral_features import clear_feature_cache, get_cache_stats
from .io.io import load_audio
from .utils.config_loader import load_config
from .utils.aggregator import aggregate_features
from .batch_extractor import extract_features_from_folder
from .methods.stft import clear_stft_cache
from .presets import load_preset, list_presets, preset_info
from .utils.config_loader import validate_config

try:
    __version__ = version('AFX')
except PackageNotFoundError:  # running from a source checkout without install
    __version__ = '0.0.0'

__all__ = [
    '__version__',
    'extract_all_features',
    'FEATURE_MAP',
    'extract_features_from_folder',
    'load_audio',
    'load_config',
    'validate_config',
    'load_preset',
    'list_presets',
    'preset_info',
    'aggregate_features',
    'clear_feature_cache',
    'get_cache_stats',
    'clear_stft_cache',
    'AFXFeatureExtractor',
]


def __getattr__(name):
    # Lazy access so `import AFX` stays light: the sklearn transformer is
    # only imported when actually requested.
    if name == 'AFXFeatureExtractor':
        from AFX.sklearn import AFXFeatureExtractor
        return AFXFeatureExtractor
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
