"""
Unified API for extracting all audio features as specified in config.
"""
from typing import Dict, Any, Optional
import numpy as np
from .extractors import time_domain, frequency_domain, cepstral_features, harmonic_features
from .utils.aggregator import aggregate_features
from .utils.shape_utils import normalize_features_dict

FEATURE_MAP = {
    'zcr': time_domain.extract_zero_crossing_rate,
    'variance': time_domain.extract_variance,
    'entropy': time_domain.extract_entropy,
    'crest_factor': time_domain.extract_crest_factor,
    'spectral_centroid': frequency_domain.extract_spectral_centroid,
    'spectral_bandwidth': frequency_domain.extract_spectral_bandwidth,
    'spectral_rolloff': frequency_domain.extract_spectral_rolloff,
    'mfcc': cepstral_features.extract_mfcc,
    'mfcc_delta': cepstral_features.extract_mfcc_delta,
    'chroma_cqt': cepstral_features.extract_chroma_cqt,
    'melspectrogram': cepstral_features.extract_melspectrogram,
    'pitch': harmonic_features.extract_pitch,
    'thd': harmonic_features.extract_thd,
    'hnr': harmonic_features.extract_hnr,
}

def extract_all_features(signal: np.ndarray, sr: int, config: Dict[str, Any], return_metadata: bool = False) -> Dict[str, np.ndarray]:
    """
    Extract all features specified in the config from the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        config: Config dict with 'features' and 'aggregation'
        return_metadata: If True, include metadata in results
    Returns:
        Dict of feature names to np.ndarray
    """
    features = {}
    metadata = {}
    for feat_name, params in config['features'].items():
        extractor = FEATURE_MAP.get(feat_name)
        if extractor is None:
            continue
        params = params.copy()
        params['return_metadata'] = return_metadata
        out = extractor(signal, sr, **params)
        if return_metadata and 'metadata' in out:
            metadata[feat_name] = out['metadata']
            features[feat_name] = out[feat_name]
        else:
            features[feat_name] = out[feat_name]
    # Aggregate if specified
    if 'aggregation' in config:
        features = aggregate_features(features, method=config['aggregation'])
    # Normalize shapes
    features = normalize_features_dict(features, mode='flatten')
    if return_metadata:
        return {'features': features, 'metadata': metadata}
    return features
