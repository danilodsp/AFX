"""
AFX: audio feature extraction for machine learning.

Top-level entry points: ``load_audio`` -> ``extract_all_features`` for one
signal, ``extract_features_from_folder`` (or the ``afx`` CLI) for a dataset,
``AFXFeatureExtractor`` for scikit-learn pipelines and ``afx_audio.torch`` for the
differentiable path. ``FEATURE_MAP`` lists every feature name.
"""
import importlib as _importlib
import importlib.metadata as _metadata

from .batch_extractor import extract_features_from_folder
from .dataset_stats import load_stats
from .extract_all import FEATURE_MAP, extract_all_features, extract_segments
from .extractors.cepstral_features import clear_feature_cache, get_cache_stats
from .io.io import load_audio
from .methods.stft import clear_stft_cache
from .presets import list_presets, load_preset, preset_info
from .provenance import config_hash, read_provenance
from .registry import feature_info, list_features, register_feature, unregister_feature
from .utils.aggregator import aggregate_features, valid_fraction
from .utils.batch_utils import feature_names, flatten_features
from .utils.config_loader import load_config, validate_config

try:
    __version__ = _metadata.version('afx-audio')
except _metadata.PackageNotFoundError:  # running from a source checkout without install
    __version__ = '0.0.0'

__all__ = [
    '__version__',
    'extract_all_features',
    'extract_segments',
    'FEATURE_MAP',
    'list_features',
    'feature_info',
    'register_feature',
    'unregister_feature',
    'feature_names',
    'flatten_features',
    'extract_features_from_folder',
    'load_audio',
    'load_config',
    'validate_config',
    'load_preset',
    'list_presets',
    'preset_info',
    'aggregate_features',
    'valid_fraction',
    'clear_feature_cache',
    'get_cache_stats',
    'clear_stft_cache',
    'config_hash',
    'read_provenance',
    'load_stats',
]

# Names resolved on first access so `import afx_audio` stays light and works
# without the optional dependencies. They stay out of __all__ so that
# `from afx_audio import *` never imports scikit-learn.
_LAZY = {
    'AFXFeatureExtractor': 'afx_audio.sklearn',
}


def __getattr__(name):
    if name in _LAZY:
        return getattr(_importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
