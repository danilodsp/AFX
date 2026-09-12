"""
Unified API for extracting all audio features as specified in config.
"""
from typing import Dict, Any
import numpy as np
from .extractors import time_domain, frequency_domain, cepstral_features, harmonic_features
from .utils.aggregator import aggregate_features


FEATURE_MAP = {
    # Time domain
    'zcr': time_domain.extract_zero_crossing_rate,
    'variance': time_domain.extract_variance,
    'entropy': time_domain.extract_entropy,
    'crest_factor': time_domain.extract_crest_factor,
    'rms_energy': time_domain.extract_rms_energy,
    'kurtosis': time_domain.extract_kurtosis,
    'short_time_energy': time_domain.extract_short_time_energy,
    'energy_ratio': time_domain.extract_energy_ratio,
    'sample_entropy': time_domain.extract_sample_entropy,
    'mobility': time_domain.extract_mobility,
    'complexity': time_domain.extract_complexity,
    'autocorr_variance': time_domain.extract_autocorrelation_variance,
    'autocorr_snr': time_domain.extract_autocorr_snr,
    'autocorr_zero_crossings': time_domain.extract_autocorr_zero_crossings,
    'autocorr_peak_width': time_domain.extract_autocorr_peak_width,
    'autocorr_decay_rate': time_domain.extract_autocorr_decay_rate,
    'autocorr_coeffs_deviation': time_domain.extract_autocorr_coeffs_deviation,
    'max_peak_autocorr_envelope': time_domain.extract_max_peak_autocorr_envelope,
    # Frequency domain
    'spectral_centroid': frequency_domain.extract_spectral_centroid,
    'spectral_bandwidth': frequency_domain.extract_spectral_bandwidth,
    'spectral_rolloff': frequency_domain.extract_spectral_rolloff,
    'spectral_contrast': frequency_domain.extract_spectral_contrast,
    'spectral_entropy': frequency_domain.extract_spectral_entropy,
    'spectral_flatness': frequency_domain.extract_spectral_flatness,
    'spectral_flux': frequency_domain.extract_spectral_flux,
    'spectral_skewness': frequency_domain.extract_spectral_skewness,
    'spectral_slope': frequency_domain.extract_spectral_slope,
    # Cepstral / spectrogram
    'mfcc': cepstral_features.extract_mfcc,
    'mfcc_delta': cepstral_features.extract_mfcc_delta,
    'mfcc_delta_delta': cepstral_features.extract_mfcc_delta_delta,
    'chroma_stft': cepstral_features.extract_chroma_stft,
    'chroma_cqt': cepstral_features.extract_chroma_cqt,
    'cqt': cepstral_features.extract_cqt,
    'melspectrogram': cepstral_features.extract_melspectrogram,
    'gfcc': cepstral_features.extract_gfcc,
    # Harmonic
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
        config: Config dict with a 'features' block mapping feature names to
            extractor keyword arguments. Optional keys: 'aggregation'
            (default 'mean'), 'flatten' (default True), 'normalize'
            ('zscore', 'minmax' or None), 'aggregation_params' (extra
            keyword arguments for ``aggregate_features``, e.g. window_size
            and hop_length for 'sliding_window'), and 'preserve_shape'
            (True skips aggregation and returns per-frame arrays).
        return_metadata: If True, include per-feature metadata (frame times)
    Returns:
        Dict of feature names to np.ndarray, or a single stacked 1-D array
        when aggregation is 'stack'. With return_metadata=True a dict with
        'features' and 'metadata' keys.
    """
    if not isinstance(config, dict) or 'features' not in config:
        raise ValueError(
            "config must be a dict with a 'features' key mapping feature "
            "names to parameter dicts (see AFX/config.json)"
        )
    signal = np.asarray(signal)
    if signal.ndim != 1:
        raise ValueError(
            f'signal must be 1-D (mono), got shape {signal.shape}. '
            'Downmix multichannel audio first, e.g. signal.mean(axis=1).'
        )
    preserve_shape = config.get('preserve_shape', False)
    features = {}
    metadata = {}
    for feat_name, params in config['features'].items():
        extractor = FEATURE_MAP.get(feat_name)
        if extractor is None:
            raise KeyError(
                f'Unknown feature: {feat_name!r}. '
                f'Available features: {sorted(FEATURE_MAP)}'
            )
        params = params.copy()
        params['return_metadata'] = return_metadata
        out = extractor(signal, sr, **params)
        if return_metadata and 'metadata' in out:
            metadata[feat_name] = out['metadata']
            features[feat_name] = out[feat_name]
        else:
            features[feat_name] = out[feat_name]

    if not preserve_shape:
        # Extra keyword arguments for aggregate_features: window_size /
        # hop_length / sliding_agg for method='sliding_window', norm_stats
        # for dataset-level normalization.
        agg_kwargs = dict(config.get('aggregation_params', {}))
        agg_kwargs['method'] = config.get('aggregation', 'mean')
        agg_kwargs['flatten'] = config.get('flatten', True)
        agg_kwargs['normalize'] = config.get('normalize', None)
        features = aggregate_features(features, **agg_kwargs)

    if return_metadata:
        return {'features': features, 'metadata': metadata}
    return features
