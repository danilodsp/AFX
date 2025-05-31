"""
Batch extractor for processing folders of audio files.
"""
import os
import numpy as np
import pandas as pd
import json
from typing import Dict, Any, List

from AFX.io.io import load_audio
from AFX.utils.config_loader import load_config
from AFX.extract_all import extract_all_features


def extract_features_from_folder(
    folder_path: str,
    config_path: str,
    output_path: str,
    file_exts: List[str] = ['.wav', '.mp3', '.flac'],
    save_format: str = 'npy'
) -> None:
    """
    Extract features from all audio files in a folder (recursively).
    Args:
        folder_path: Root folder to search for audio files
        config_path: Path to config.json
        output_path: Path to save output (npy, json, or csv)
        file_exts: List of file extensions to include
        save_format: Output format ('npy', 'json', 'csv')
    """
    config = load_config(config_path)
    results = {}
    for root, _, files in os.walk(folder_path):
        for fname in files:
            if not any(fname.lower().endswith(ext) for ext in file_exts):
                continue
            fpath = os.path.join(root, fname)
            try:
                signal, sr = load_audio(fpath, sr=config['sample_rate'])
                feats = extract_all_features(signal, sr, config)
                results[fname] = {k: v.tolist() for k, v in feats.items()}
            except Exception as e:
                results[fname] = {'error': str(e)}
    if save_format == 'npy':
        np.save(output_path, results)
    elif save_format == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
    elif save_format == 'csv':
        df = pd.DataFrame.from_dict(results, orient='index')
        df.to_csv(output_path)
    else:
        raise ValueError(f'Unknown save_format: {save_format}')
