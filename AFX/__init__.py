# __init__.py for AFX package

from .extract_all import extract_all_features, FEATURE_MAP
from .extractors.cepstral_features import clear_feature_cache, get_cache_stats
from .io.io import load_audio
from .utils.config_loader import load_config
from .utils.aggregator import aggregate_features
from .batch_extractor import extract_features_from_folder
from .methods.stft import clear_stft_cache
from .presets import load_preset, list_presets, preset_info

__all__ = [
    'extract_all_features',
    'FEATURE_MAP',
    'extract_features_from_folder',
    'load_audio',
    'load_config',
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
