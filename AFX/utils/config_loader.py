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
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Basic validation
    if 'sample_rate' not in config:
        raise ValueError('Config must specify sample_rate')
    if 'features' not in config:
        raise ValueError('Config must specify features block')
    return config
