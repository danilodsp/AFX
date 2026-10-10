"""
Configuration loader for AFX (JSON schema).
"""
import difflib
import inspect
import json
import numbers
import os
import re
from typing import Any

VALID_AGGREGATIONS = (
    'mean', 'std', 'summary', 'advanced_summary', 'skew', 'kurtosis',
    'sliding_window', 'stack',
)
VALID_NORMALIZATIONS = ('zscore', 'minmax', None)
CONFIG_KEYS = (
    'sample_rate', 'features', 'aggregation', 'aggregation_params', 'flatten', 'normalize', 'preserve_shape',
    'coverage',
    # descriptive fields of the presets
    'name', 'title', 'description', 'preset',
)
PRESET_NAME = re.compile(r'[a-z][a-z0-9_]*')
AGGREGATION_PARAMS = ('window_size', 'hop_length', 'sliding_agg', 'norm_stats')

# Arguments every extractor takes that are not configurable per feature
_RESERVED = ('signal', 'sr', 'return_metadata')
# Integer parameters that must be positive when given
_POSITIVE_INT = ('frame_size', 'hop_length', 'n_fft', 'n_mfcc', 'n_mels', 'n_gfcc', 'n_gammatone',
                 'n_bins', 'bins_per_octave', 'n_bands', 'max_lag', 'num_bins', 'n_harmonics', 'width')


def integral_rate(value: Any) -> Any:
    """16000.0 becomes 16000, so a rate read from JSON or given as a float hashes like the int."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def feature_parameters(name: str) -> dict[str, Any]:
    """Configurable parameters of a feature and their defaults."""
    from afx_audio.extract_all import FEATURE_MAP

    sig = inspect.signature(FEATURE_MAP[name])
    return {
        p.name: p.default for p in sig.parameters.values()
        if p.name not in _RESERVED and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }


def _suggest(name: str, options) -> str:
    close = difflib.get_close_matches(name, list(options), n=1, cutoff=0.6)
    return f' Did you mean {close[0]!r}?' if close else ''


def check_feature_params(name: str, params: dict[str, Any]) -> None:
    """Raise ValueError for parameters the extractor does not accept or obviously bad values."""
    from afx_audio.extract_all import FEATURE_MAP

    accepted = feature_parameters(name)
    open_ended = any(p.kind == p.VAR_KEYWORD for p in inspect.signature(FEATURE_MAP[name]).parameters.values())
    for key in params:
        if key not in accepted and not open_ended:
            raise ValueError(
                f'Unknown parameter {key!r} for feature {name!r}.{_suggest(key, accepted)} '
                f'Accepted: {sorted(accepted)}'
            )
    for key in _POSITIVE_INT:
        value = params.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, numbers.Integral) or value <= 0:
            raise ValueError(f'{name}.{key} must be a positive integer, got {value!r}')


def unknown_feature_error(name: str) -> ValueError:
    """The error for a feature name with no extractor, with the closest match if there is one."""
    from afx_audio.extract_all import FEATURE_MAP

    return ValueError(
        f'Unknown feature {name!r}.{_suggest(name, FEATURE_MAP)} '
        'afx_audio.list_features() or `afx --list-features` lists them.'
    )


def validate_config(config: dict[str, Any], require_sample_rate: bool = True) -> dict[str, Any]:
    """
    Check a config dict and return it.

    ``features`` may be a list of names as a shorthand for default
    parameters, and ``sample_rate`` a float with an integral value; the
    returned config is then a copy with the list expanded to {name: {}}
    and the rate an int. Otherwise the config itself is returned.

    Raises ValueError for unknown top-level keys, a missing (unless
    ``require_sample_rate`` is False) or non-positive sample_rate, a
    missing features block, feature names with no extractor, parameters
    the extractor does not accept, non-positive frame sizes and counts, an
    unknown aggregation or normalize value, sliding_window aggregation
    without window_size and hop_length in aggregation_params, and a
    non-boolean coverage or coverage with 'stack' or preserve_shape, and
    a preset block other than {"name": <lowercase name>, "version": <int >= 1>}.
    """
    from afx_audio.extract_all import FEATURE_MAP

    if not isinstance(config, dict):
        raise ValueError('Config root must be a JSON object')
    for key in config:
        if key not in CONFIG_KEYS:
            raise ValueError(f'Unknown config key {key!r}.{_suggest(key, CONFIG_KEYS)} Valid keys: {CONFIG_KEYS}')
    if 'sample_rate' not in config:
        if require_sample_rate:
            raise ValueError('Config must specify sample_rate')
    elif not isinstance(config['sample_rate'], (int, float)) or isinstance(config['sample_rate'], bool):
        raise ValueError(
            f"sample_rate must be a number, got {type(config['sample_rate']).__name__}"
        )
    elif config['sample_rate'] <= 0:
        raise ValueError(f"sample_rate must be positive, got {config['sample_rate']}")
    elif integral_rate(config['sample_rate']) is not config['sample_rate']:
        config = {**config, 'sample_rate': integral_rate(config['sample_rate'])}
    features = config.get('features')
    if isinstance(features, (list, tuple)):
        if not all(isinstance(n, str) for n in features):
            raise ValueError('A features list must contain feature names only')
        config = {**config, 'features': {name: {} for name in features}}
    elif not isinstance(features, dict):
        raise ValueError(
            "Config must have a 'features' block: an object mapping feature names to parameters, "
            "or a list of names, e.g. {'features': ['mfcc', 'zcr']}"
        )
    for name, params in config['features'].items():
        if name not in FEATURE_MAP:
            raise unknown_feature_error(name)
        if not isinstance(params, dict):
            raise ValueError(
                f'Parameters for feature {name!r} must be a JSON object, '
                f'got {type(params).__name__}'
            )
        check_feature_params(name, params)
    aggregation = config.get('aggregation', 'mean')
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f'Unknown aggregation {aggregation!r}.{_suggest(str(aggregation), VALID_AGGREGATIONS)} '
            f'Valid: {VALID_AGGREGATIONS}'
        )
    agg_params = config.get('aggregation_params', {})
    if not isinstance(agg_params, dict):
        raise ValueError('aggregation_params must be a JSON object')
    for key in agg_params:
        if key not in AGGREGATION_PARAMS:
            raise ValueError(f'Unknown aggregation_params key {key!r}.{_suggest(key, AGGREGATION_PARAMS)}')
    if aggregation == 'sliding_window' and not {'window_size', 'hop_length'} <= set(agg_params):
        raise ValueError("aggregation 'sliding_window' needs window_size and hop_length (in frames) "
                         'under aggregation_params')
    normalize = config.get('normalize')
    if normalize not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalize: {normalize!r}. Valid: 'zscore', 'minmax' or null"
        )
    coverage = config.get('coverage', False)
    if not isinstance(coverage, bool):
        raise ValueError(f'coverage must be true or false, got {coverage!r}')
    if coverage and (aggregation == 'stack' or config.get('preserve_shape')):
        raise ValueError(
            "coverage adds <feature>_valid_frac to aggregated output; it does not apply with aggregation 'stack' "
            'or preserve_shape. Use afx_audio.utils.aggregator.valid_fraction on per-frame arrays instead.'
        )
    if 'preset' in config:
        check_preset_block(config['preset'])
    return config


def check_preset_block(block: Any) -> None:
    """Raise ValueError unless ``block`` is {'name': <lowercase name>, 'version': <positive int>}."""
    if not isinstance(block, dict) or set(block) != {'name', 'version'}:
        raise ValueError(f'preset must be an object with exactly the keys name and version, got {block!r}')
    name, version = block['name'], block['version']
    if not isinstance(name, str) or not PRESET_NAME.fullmatch(name):
        raise ValueError(f'preset name must be lowercase letters, digits and underscores, got {name!r}')
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError(f'preset version must be a positive integer, got {version!r}')


def load_config(config_path: str) -> dict[str, Any]:
    """
    Load and validate a JSON config file for feature extraction.
    Args:
        config_path: Path to config.json
    Returns:
        Config dictionary (see validate_config for the checks applied)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in config file {config_path}: {e}') from e
    if not isinstance(config, dict):
        raise ValueError(f'Config root must be a JSON object: {config_path}')
    return validate_config(config)
