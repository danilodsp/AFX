"""
Configuration loader for AFX (JSON schema).
"""
from typing import Dict, Any
import json
import os

VALID_AGGREGATIONS = (
    'mean', 'std', 'summary', 'advanced_summary', 'skew', 'kurtosis',
    'sliding_window', 'stack',
)
VALID_NORMALIZATIONS = ('zscore', 'minmax', None)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Check a config dict and return it unchanged.

    Raises ValueError for a missing or non-numeric sample_rate, a missing
    features block, feature names with no extractor, non-dict feature
    parameters, or an unknown aggregation / normalize value.
    """
    # Imported here: extract_all imports the extractors, which import utils
    from AFX.extract_all import FEATURE_MAP

    if not isinstance(config, dict):
        raise ValueError('Config root must be a JSON object')
    if 'sample_rate' not in config:
        raise ValueError('Config must specify sample_rate')
    if not isinstance(config['sample_rate'], (int, float)) or isinstance(
        config['sample_rate'], bool
    ):
        raise ValueError(
            f"sample_rate must be a number, got {type(config['sample_rate']).__name__}"
        )
    if 'features' not in config or not isinstance(config['features'], dict):
        raise ValueError('Config must specify a features block (JSON object)')
    unknown = [name for name in config['features'] if name not in FEATURE_MAP]
    if unknown:
        raise ValueError(
            f'Unknown feature(s) in config: {unknown}. '
            f'Available: {sorted(FEATURE_MAP)}'
        )
    for name, params in config['features'].items():
        if not isinstance(params, dict):
            raise ValueError(
                f'Parameters for feature {name!r} must be a JSON object, '
                f'got {type(params).__name__}'
            )
    aggregation = config.get('aggregation', 'mean')
    if aggregation not in VALID_AGGREGATIONS:
        raise ValueError(
            f'Unknown aggregation: {aggregation!r}. Valid: {VALID_AGGREGATIONS}'
        )
    normalize = config.get('normalize')
    if normalize not in VALID_NORMALIZATIONS:
        raise ValueError(
            f"Unknown normalize: {normalize!r}. Valid: 'zscore', 'minmax' or null"
        )
    return config


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a JSON config file for feature extraction.
    Args:
        config_path: Path to config.json
    Returns:
        Config dictionary (see validate_config for the checks applied)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in config file {config_path}: {e}') from e
    if not isinstance(config, dict):
        raise ValueError(f'Config root must be a JSON object: {config_path}')
    return validate_config(config)
