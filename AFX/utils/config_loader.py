"""
Configuration loader for AFX (JSON schema).
"""
from typing import Dict, Any
import json
import os

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate a JSON config file for feature extraction.
    Args:
        config_path: Path to config.json
    Returns:
        Config dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r', encoding='utf-8') as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in config file {config_path}: {e}') from e
    # Basic validation
    if not isinstance(config, dict):
        raise ValueError(f'Config root must be a JSON object: {config_path}')
    if 'sample_rate' not in config:
        raise ValueError('Config must specify sample_rate')
    if not isinstance(config['sample_rate'], (int, float)):
        raise ValueError(
            f"sample_rate must be a number, got {type(config['sample_rate']).__name__}"
        )
    if 'features' not in config or not isinstance(config['features'], dict):
        raise ValueError('Config must specify a features block (JSON object)')
    return config
