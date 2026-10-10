"""
Unified API for extracting all audio features as specified in config.
"""
import os
import warnings
from collections.abc import Callable
from typing import Any, cast

import numpy as np

from .extractors import cepstral_features, egemaps, frequency_domain, harmonic_features, linear_prediction, time_domain
from .utils.aggregator import aggregate_features
from .utils.config_loader import validate_config
from .utils.validation import validate_signal, validate_sr

FEATURE_MAP: dict[str, Callable[..., dict[str, Any]]] = {
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
    'spectral_kurtosis': frequency_domain.extract_spectral_kurtosis,
    'spectral_crest': frequency_domain.extract_spectral_crest,
    'spectral_decrease': frequency_domain.extract_spectral_decrease,
    # Cepstral / spectrogram
    'mfcc': cepstral_features.extract_mfcc,
    'mfcc_delta': cepstral_features.extract_mfcc_delta,
    'mfcc_delta_delta': cepstral_features.extract_mfcc_delta_delta,
    'chroma_stft': cepstral_features.extract_chroma_stft,
    'chroma_cqt': cepstral_features.extract_chroma_cqt,
    'cqt': cepstral_features.extract_cqt,
    'melspectrogram': cepstral_features.extract_melspectrogram,
    'gfcc': cepstral_features.extract_gfcc,
    'tonnetz': cepstral_features.extract_tonnetz,
    'onset_strength': cepstral_features.extract_onset_strength,
    # Harmonic
    'pitch': harmonic_features.extract_pitch,
    'thd': harmonic_features.extract_thd,
    'hnr': harmonic_features.extract_hnr,
    # Linear prediction
    'lpc': linear_prediction.extract_lpc,
    'formants': linear_prediction.extract_formants,
    # Parameter sets
    'egemaps_lld': egemaps.extract_egemaps_lld,
    'egemaps': egemaps.extract_egemaps,
}

def extract_all_features(
    signal: np.ndarray,
    sr: int,
    config: dict[str, Any],
    return_metadata: bool = False
) -> dict[str, Any] | np.ndarray:
    """
    Extract all features specified in the config from the audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        config: Config dict with a 'features' block mapping feature names to
            extractor keyword arguments (or a list of names for default
            parameters). Optional keys: 'aggregation'
            (default 'mean'), 'flatten' (default True), 'normalize'
            ('zscore', 'minmax' or None), 'aggregation_params' (extra
            keyword arguments for ``aggregate_features``, e.g. window_size
            and hop_length for 'sliding_window'), 'preserve_shape'
            (True skips aggregation and returns per-frame arrays) and
            'coverage' (True adds '<name>_valid_frac', the fraction of
            frames with a finite value, after each feature's statistics).
        return_metadata: If True, include per-feature metadata (frame times)
    Returns:
        Dict of feature names to np.ndarray, or a single stacked 1-D array
        when aggregation is 'stack'. With return_metadata=True a dict with
        'features' and 'metadata' keys.
    """
    if isinstance(signal, (str, os.PathLike)) or (
        isinstance(signal, (list, tuple)) and signal and isinstance(signal[0], (str, os.PathLike))
    ):
        raise TypeError(
            'extract_all_features takes a signal array. Load a file with afx_audio.load_audio; for many files use '
            'afx_audio.extract_features_from_folder or afx_audio.sklearn.AFXFeatureExtractor.'
        )
    signal = validate_signal(signal)
    validate_sr(sr)
    # sr is passed separately, so the config's own sample_rate is optional here
    config = validate_config(config, require_sample_rate=False)
    if 'sample_rate' in config and config['sample_rate'] != sr:
        warnings.warn(
            f"sr={sr} differs from the config's sample_rate={config['sample_rate']}. Frame sizes are in samples, "
            'so the analysis windows change length in time; resample with load_audio(path, sr=config["sample_rate"]).',
            stacklevel=2,
        )

    preserve_shape = config.get('preserve_shape', False)
    per_frame: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for feat_name, params in config['features'].items():
        out = FEATURE_MAP[feat_name](signal, sr, return_metadata=return_metadata, **params)
        per_frame[feat_name] = out[feat_name]
        if return_metadata and 'metadata' in out:
            metadata[feat_name] = out['metadata']

    features: dict[str, np.ndarray] | np.ndarray = per_frame

    if not preserve_shape:
        # Extra keyword arguments for aggregate_features: window_size /
        # hop_length / sliding_agg for method='sliding_window', norm_stats
        # for dataset-level normalization.
        agg_kwargs = dict(config.get('aggregation_params', {}))
        agg_kwargs['method'] = config.get('aggregation', 'mean')
        agg_kwargs['flatten'] = config.get('flatten', True)
        agg_kwargs['normalize'] = config.get('normalize', None)
        agg_kwargs['coverage'] = config.get('coverage', False)
        features = aggregate_features(per_frame, **agg_kwargs)

    if return_metadata:
        return {'features': features, 'metadata': metadata}
    return features


def extract_segments(
    signal: np.ndarray,
    sr: int,
    config: dict[str, Any],
    segment_length: float,
    hop: float | None = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    One aggregated feature vector per fixed-length segment of a long signal.

    The signal is cut into segments of ``segment_length`` seconds every
    ``hop`` seconds (default: no overlap; a trailing partial segment is
    dropped, and a signal shorter than one segment is a single segment),
    and ``extract_all_features`` runs on each with ``config``.

    Returns:
        (names, X, starts): column names as in flatten_features, an array
        of shape (n_segments, n_features), and each segment's start time
        in seconds.
    """
    from afx_audio.utils.batch_utils import flatten_features

    if segment_length <= 0 or (hop is not None and hop <= 0):
        raise ValueError('segment_length and hop must be positive')
    if config.get('preserve_shape') or config.get('aggregation') == 'stack':
        raise ValueError("extract_segments needs an aggregation other than 'stack' and no preserve_shape")
    signal = validate_signal(signal)
    validate_sr(sr)
    seg = max(1, int(round(segment_length * sr)))
    step = seg if hop is None else max(1, int(round(hop * sr)))
    starts = np.arange(0, max(len(signal) - seg, 0) + 1, step)
    names: list[str] = []
    rows = []
    for start in starts:
        feats = cast(dict[str, Any], extract_all_features(signal[start:start + seg], sr, config))
        seg_names, values = flatten_features(feats)
        if names and seg_names != names:
            raise ValueError('feature dimensions changed between segments; fix the frame sizes in the config')
        names = seg_names
        rows.append(values)
    return names, np.vstack(rows), starts / float(sr)
