"""
Descriptions of the built-in features: group, units, output shape and parameters.
"""
import functools
import inspect
from collections.abc import Callable
from typing import Any

from afx_audio.extract_all import FEATURE_MAP
from afx_audio.utils.config_loader import feature_parameters

# name: (group, units, rows per frame)
_INFO = {
    'zcr': ('time', 'crossings per sample', '1'),
    'variance': ('time', 'amplitude^2', '1'),
    'entropy': ('time', 'bits', '1'),
    'crest_factor': ('time', 'ratio', '1'),
    'rms_energy': ('time', 'amplitude', '1'),
    'kurtosis': ('time', 'dimensionless', '1'),
    'short_time_energy': ('time', 'amplitude^2', '1'),
    'energy_ratio': ('time', 'ratio', '1'),
    'sample_entropy': ('time', 'nats', '1'),
    'mobility': ('time', 'ratio', '1'),
    'complexity': ('time', 'ratio', '1'),
    'autocorr_variance': ('autocorrelation', 'amplitude^4', '1'),
    'autocorr_snr': ('autocorrelation', 'ratio', '1'),
    'autocorr_zero_crossings': ('autocorrelation', 'count', '1'),
    'autocorr_peak_width': ('autocorrelation', 'lags', '1'),
    'autocorr_decay_rate': ('autocorrelation', 'log units per lag', '1'),
    'autocorr_coeffs_deviation': ('autocorrelation', 'amplitude^2', '1'),
    'max_peak_autocorr_envelope': ('autocorrelation', 'amplitude^2', '1'),
    'spectral_centroid': ('spectral', 'Hz', '1'),
    'spectral_bandwidth': ('spectral', 'Hz', '1'),
    'spectral_rolloff': ('spectral', 'Hz', '1'),
    'spectral_contrast': ('spectral', 'dB', 'n_bands + 1'),
    'spectral_entropy': ('spectral', 'bits', '1'),
    'spectral_flatness': ('spectral', 'ratio', '1'),
    'spectral_flux': ('spectral', 'magnitude', '1'),
    'spectral_skewness': ('spectral', 'dimensionless', '1'),
    'spectral_slope': ('spectral', 'dB per Hz', '1'),
    'spectral_kurtosis': ('spectral', 'dimensionless', '1'),
    'spectral_crest': ('spectral', 'ratio', '1'),
    'spectral_decrease': ('spectral', 'per bin', '1'),
    'melspectrogram': ('spectrogram', 'power', 'n_mels'),
    'cqt': ('spectrogram', 'magnitude', 'n_bins'),
    'chroma_stft': ('spectrogram', 'normalized energy', '12'),
    'chroma_cqt': ('spectrogram', 'normalized magnitude', '12'),
    'tonnetz': ('spectrogram', 'dimensionless', '6'),
    'onset_strength': ('spectrogram', 'dB', '1'),
    'mfcc': ('cepstral', 'dB (DCT of log-mel)', 'n_mfcc'),
    'mfcc_delta': ('cepstral', 'dB per frame', 'n_mfcc'),
    'mfcc_delta_delta': ('cepstral', 'dB per frame^2', 'n_mfcc'),
    'gfcc': ('cepstral', 'dB (DCT of log-gammatone)', 'n_gfcc'),
    'pitch': ('harmonic', 'Hz (NaN = unvoiced)', '1'),
    'thd': ('harmonic', 'ratio', '1'),
    'hnr': ('harmonic', 'dB', '1'),
    'lpc': ('linear prediction', 'dimensionless', 'order'),
    'formants': ('linear prediction', 'Hz', 'n_formants'),
    'egemaps_lld': ('parameter set', 'per row, see EGEMAPS_LLD_NAMES', '25'),
    'egemaps': ('parameter set', 'per row, see EGEMAPS_NAMES', '88'),
}

GROUPS = ('time', 'autocorrelation', 'spectral', 'spectrogram', 'cepstral', 'harmonic', 'linear prediction',
          'parameter set', 'custom')

# When a built-in feature is NaN on a frame. Features not listed are finite
# on every frame of valid input (silence gives 0 or the floor of a log).
_UNDEFINED = {
    'kurtosis': 'constant frame (zero variance), including digital silence',
    'sample_entropy': 'no template matches (A or B is 0), or fewer than m + 2 samples',
    'mobility': 'frame shorter than 2 samples',
    'complexity': 'frame shorter than 3 samples',
    'autocorr_decay_rate': 'fewer than 3 autocorrelation lags (max_lag or frame below 3)',
    'max_peak_autocorr_envelope': 'only lag 0 available (max_lag or frame of 1)',
    'spectral_kurtosis': 'silent frame or zero spectral variance (single bin)',
    'pitch': 'unvoiced frames (no YIN dip below threshold and minimum not below 0.5) and f0 outside [fmin, fmax]',
    'thd': 'zero-energy frame or fewer than 2 FFT bins',
    'hnr': 'zero-energy frame; every frame when sr / fmax >= min(sr / fmin, frame_size - 1)',
    'lpc': 'frame not longer than order',
    'formants': 'fewer than n_formants LPC resonances above fmin and narrower than max_bandwidth '
                '(silence, noise, unvoiced sounds), or frame not longer than order',
    'egemaps_lld': 'voice-dependent rows (F0, jitter, shimmer, HNR, formants, H1-H2, H1-A3) on unvoiced frames',
    'egemaps': 'statistics over voiced or unvoiced segments when the clip has none',
}


def list_features(group: str | None = None) -> list[str]:
    """Feature names, optionally only those of one group (see GROUPS)."""
    if group is not None and group not in GROUPS:
        raise ValueError(f'Unknown group {group!r}. Groups: {GROUPS}')
    return [name for name in FEATURE_MAP if group is None or _INFO[name][0] == group]


def feature_info(name: str) -> dict[str, Any]:
    """
    Group, units, output rows, parameters with defaults and a one-line summary for a feature.

    ``rows`` is the size of the first axis of the per-frame output ('1'
    means the output is 1-D, one value per frame). ``undefined`` says when
    a frame is NaN, None if never (or unknown, for custom features).
    """
    if name not in FEATURE_MAP:
        raise KeyError(f'Unknown feature: {name!r}. Available: {sorted(FEATURE_MAP)}')
    group, units, rows = _INFO[name]
    func = FEATURE_MAP[name]
    doc = (func.__doc__ or '').strip()
    summary = ' '.join(doc.split('\n\n')[0].split())
    return {
        'name': name,
        'group': group,
        'units': units,
        'rows': rows,
        'parameters': feature_parameters(name),
        'summary': summary,
        'function': f'{func.__module__}.{func.__name__}',
        'undefined': None if group == 'custom' else _UNDEFINED.get(name),
    }


# User registrations: name -> (function as given, units, rows), so worker
# processes can repeat them, and the built-in entries they replaced
_CUSTOM: dict[str, tuple[Callable[..., dict[str, Any]], str, str]] = {}
_REPLACED: dict[str, tuple[Callable[..., dict[str, Any]], tuple[str, str, str]]] = {}


def register_feature(
    name: str,
    func: Callable[..., dict[str, Any]],
    *,
    units: str = 'unspecified',
    rows: str = '1',
    overwrite: bool = False,
) -> None:
    """
    Make ``func`` available under ``name`` in configs, FEATURE_MAP and feature_info.

    ``func(signal, sr, **params)`` must return ``{name: array}`` with the
    per-frame values on the last axis (or a single value). It may accept
    ``return_metadata`` and add ``{'metadata': {'times': ...}}``; if it
    does not, the argument is dropped for it. Its keyword parameters
    become the feature's configurable parameters (any parameter is
    accepted if it takes ``**kwargs``). The batch extractor and
    AFXFeatureExtractor pass registrations on to their worker processes,
    which needs ``func`` to be picklable: defined at module level, not a
    lambda or a nested function.
    """
    if name in FEATURE_MAP and not overwrite:
        raise ValueError(f'Feature {name!r} already exists; pass overwrite=True to replace it')
    if name in FEATURE_MAP and name not in _CUSTOM and name not in _REPLACED:
        _REPLACED[name] = (FEATURE_MAP[name], _INFO[name])
    params = inspect.signature(func).parameters
    if 'return_metadata' in params or any(p.kind == p.VAR_KEYWORD for p in params.values()):
        extractor = func
    else:
        @functools.wraps(func)
        def extractor(signal, sr, return_metadata=False, **kwargs):
            return func(signal, sr, **kwargs)
        sig = inspect.signature(func)
        extractor.__signature__ = sig.replace(parameters=[  # type: ignore[attr-defined]
            *sig.parameters.values(),
            inspect.Parameter('return_metadata', inspect.Parameter.KEYWORD_ONLY, default=False),
        ])
    FEATURE_MAP[name] = extractor
    _INFO[name] = ('custom', units, rows)
    _CUSTOM[name] = (func, units, rows)


def unregister_feature(name: str) -> None:
    """Remove a feature added with register_feature, restoring the built-in one it replaced."""
    if name not in _CUSTOM:
        raise KeyError(f'{name!r} is not a registered custom feature')
    del _CUSTOM[name]
    if name in _REPLACED:
        FEATURE_MAP[name], _INFO[name] = _REPLACED.pop(name)
    else:
        del FEATURE_MAP[name]
        del _INFO[name]


def custom_features() -> dict[str, tuple[Callable[..., dict[str, Any]], str, str]]:
    """The current registrations, for handing to worker processes."""
    return dict(_CUSTOM)


def restore_custom_features(custom: dict[str, tuple[Callable[..., dict[str, Any]], str, str]]) -> None:
    """Repeat registrations from custom_features() in a worker process."""
    for name, (func, units, rows) in custom.items():
        if _CUSTOM.get(name, (None,))[0] is not func:
            register_feature(name, func, units=units, rows=rows, overwrite=True)
