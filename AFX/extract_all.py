"""Unified API for extracting all audio features as specified in config."""
import logging
from typing import Dict, Any
import numpy as np
from .extractors import time_domain, frequency_domain, cepstral_features, harmonic_features
from .utils.aggregator import aggregate_features

logger = logging.getLogger(__name__)


FEATURE_MAP = {
    # Time-domain features
    'zcr': time_domain.extract_zero_crossing_rate,
    'variance': time_domain.extract_variance,
    'entropy': time_domain.extract_entropy,
    'crest_factor': time_domain.extract_crest_factor,
    'rms_energy': time_domain.extract_rms_energy,
    'kurtosis': time_domain.extract_kurtosis,
    'autocorr_variance': time_domain.extract_autocorrelation_variance,
    'mobility': time_domain.extract_mobility,
    'complexity': time_domain.extract_complexity,
    'sample_entropy': time_domain.extract_sample_entropy,
    'energy_ratio': time_domain.extract_energy_ratio,
    'short_time_energy': time_domain.extract_short_time_energy,
    # Frequency-domain features
    'spectral_centroid': frequency_domain.extract_spectral_centroid,
    'spectral_bandwidth': frequency_domain.extract_spectral_bandwidth,
    'spectral_rolloff': frequency_domain.extract_spectral_rolloff,
    'spectral_contrast': frequency_domain.extract_spectral_contrast,
    'spectral_entropy': frequency_domain.extract_spectral_entropy,
    'spectral_flatness': frequency_domain.extract_spectral_flatness,
    'spectral_flux': frequency_domain.extract_spectral_flux,
    'spectral_skewness': frequency_domain.extract_spectral_skewness,
    'spectral_slope': frequency_domain.extract_spectral_slope,
    # Cepstral / chroma features
    'mfcc': cepstral_features.extract_mfcc,
    'mfcc_delta': cepstral_features.extract_mfcc_delta,
    'mfcc_delta_delta': cepstral_features.extract_mfcc_delta_delta,
    'chroma_cqt': cepstral_features.extract_chroma_cqt,
    'chroma_stft': cepstral_features.extract_chroma_stft,
    'cqt': cepstral_features.extract_cqt,
    'melspectrogram': cepstral_features.extract_melspectrogram,
    'gfcc': cepstral_features.extract_gfcc,
    # Harmonic features
    'pitch': harmonic_features.extract_pitch,
    'thd': harmonic_features.extract_thd,
    'hnr': harmonic_features.extract_hnr,
}

def extract_all_features(
    signal: np.ndarray,
    sr: int,
    config: Dict[str, Any],
    return_metadata: bool = False
) -> Dict[str, np.ndarray]:
    """
    Extract all features specified in the config from the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        config: Config dict with 'features' and 'aggregation'. Shape
            preservation is controlled via ``config['preserve_shape']``.
        return_metadata: If True, include metadata in results
    Returns:
        Dict of feature names to np.ndarray
    """
    preserve_shape = config.get('preserve_shape', False)
    features = {}
    metadata = {}
    for feat_name, params in config['features'].items():
        extractor = FEATURE_MAP.get(feat_name)
        if extractor is None:
            logger.warning("Feature '%s' in config has no matching extractor and will be skipped.", feat_name)
            continue
        params = params.copy()
        params['return_metadata'] = return_metadata
        out = extractor(signal, sr, **params)
        if return_metadata and 'metadata' in out:
            metadata[feat_name] = out['metadata']
            features[feat_name] = out[feat_name]
        else:
            features[feat_name] = out[feat_name]

    if not preserve_shape:
        # Aggregate, flatten, and normalize if specified
        agg_kwargs = {}
        # Always pass aggregation, flatten, and normalize, with sensible defaults
        agg_kwargs['method'] = config.get('aggregation', 'mean')
        agg_kwargs['flatten'] = config.get('flatten', True)
        agg_kwargs['normalize'] = config.get('normalize', None)
        features = aggregate_features(features, **agg_kwargs)
    # else: keep original shapes

    if return_metadata:
        return {'features': features, 'metadata': metadata}
    return features
