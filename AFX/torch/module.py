"""
``AFXFeatureModule`` — an ``nn.Module`` wrapper around the differentiable
AFX torch transforms, so feature extraction can sit inside a training
graph (e.g. as the front end of a classifier, or for feature-space losses).
"""
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from AFX.torch import functional as F_afx

# Feature name -> (functional, params that need precomputed constants)
_TORCH_FEATURES = {
    'melspectrogram': F_afx.melspectrogram,
    'mfcc': F_afx.mfcc,
    'spectral_centroid': F_afx.spectral_centroid,
    'spectral_bandwidth': F_afx.spectral_bandwidth,
    'spectral_rolloff': F_afx.spectral_rolloff,
    'spectral_entropy': F_afx.spectral_entropy,
    'spectral_flatness': F_afx.spectral_flatness,
    'spectral_flux': F_afx.spectral_flux,
    'zcr': F_afx.zero_crossing_rate,
    'rms_energy': F_afx.rms_energy,
}

# Params accepted by each functional (anything else in the config entry is
# ignored with the same tolerance as the NumPy extractors' **kwargs).
_ACCEPTED_PARAMS = {
    'melspectrogram': {'n_mels', 'frame_size', 'hop_length', 'fmin', 'fmax',
                       'center'},
    'mfcc': {'n_mfcc', 'frame_size', 'hop_length', 'n_mels', 'fmin', 'fmax',
             'center'},
    'spectral_centroid': {'frame_size', 'hop_length', 'center'},
    'spectral_bandwidth': {'frame_size', 'hop_length', 'center'},
    'spectral_rolloff': {'frame_size', 'hop_length', 'roll_percent',
                         'center'},
    'spectral_entropy': {'frame_size', 'hop_length', 'center'},
    'spectral_flatness': {'frame_size', 'hop_length', 'center'},
    'spectral_flux': {'frame_size', 'hop_length', 'center'},
    'zcr': {'frame_size', 'hop_length'},
    'rms_energy': set(),
}

_SPECTRAL = set(_TORCH_FEATURES) - {'zcr', 'rms_energy'}


def supported_features() -> List[str]:
    """Names of features available on the torch path."""
    return sorted(_TORCH_FEATURES)


class AFXFeatureModule(nn.Module):
    """
    Extract a configurable set of AFX features as torch tensors.

    Mirrors the ``extract_all_features`` config convention: ``features``
    maps feature names to parameter dicts. Constant tensors (Hann windows,
    mel filterbanks, DCT matrices) are precomputed in ``__init__`` and
    registered as buffers, so the module moves correctly with
    ``.to(device)`` and adds no per-forward construction overhead.

    Args:
        features: Dict like ``config['features']``, e.g.
            ``{'mfcc': {'n_mfcc': 13, 'frame_size': 2048, 'hop_length': 512},
               'spectral_centroid': {'frame_size': 2048, 'hop_length': 512}}``.
            Only torch-supported features are allowed (see
            ``supported_features()``).
        sample_rate: Audio sample rate the module expects.
        aggregation: 'none' (keep per-frame values), 'mean', 'std', or
            'mean_std' (both, concatenated) over the time axis.
        output: 'dict' returns ``{name: tensor}``; 'vector' flattens and
            concatenates everything into a single (batch, dim) tensor
            (features sorted by insertion order, matching
            ``feature_dim()`` / ``feature_names()``).

    Example::

        module = AFXFeatureModule(
            features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
            sample_rate=16000, aggregation='mean', output='vector',
        ).to('cuda')
        feats = module(waveforms)          # (batch, dim), differentiable
    """

    def __init__(
        self,
        features: Dict[str, Dict[str, Any]],
        sample_rate: int = 22050,
        aggregation: str = 'none',
        output: str = 'dict',
    ):
        super().__init__()
        if not features:
            raise ValueError('features must be a non-empty dict')
        unknown = sorted(set(features) - set(_TORCH_FEATURES))
        if unknown:
            raise ValueError(
                f'Feature(s) not available on the torch path: {unknown}. '
                f'Supported: {supported_features()}. For other features '
                f'use the NumPy path (AFX.extract_all_features).'
            )
        if aggregation not in ('none', 'mean', 'std', 'mean_std'):
            raise ValueError(
                "aggregation must be 'none', 'mean', 'std' or 'mean_std'"
            )
        if output not in ('dict', 'vector'):
            raise ValueError("output must be 'dict' or 'vector'")

        self.sample_rate = sample_rate
        self.aggregation = aggregation
        self.output = output
        self._feature_params: Dict[str, Dict[str, Any]] = {}

        for name, params in features.items():
            params = dict(params or {})
            accepted = _ACCEPTED_PARAMS[name]
            params = {k: v for k, v in params.items() if k in accepted}
            self._feature_params[name] = params

            # Precompute constants as buffers
            if name in _SPECTRAL:
                frame_size = params.get('frame_size', 2048)
                self.register_buffer(
                    f'_window_{name}',
                    F_afx.hann_window(frame_size),
                    persistent=False,
                )
            if name in ('melspectrogram', 'mfcc'):
                frame_size = params.get('frame_size', 2048)
                n_mels = params.get('n_mels', 128)
                fmin = params.get('fmin', 0.0)
                fmax = params.get('fmax', None)
                self.register_buffer(
                    f'_melfb_{name}',
                    F_afx.mel_filterbank(
                        n_mels, frame_size, sr=sample_rate,
                        fmin=fmin, fmax=fmax,
                    ),
                    persistent=False,
                )
            if name == 'mfcc':
                n_mfcc = params.get('n_mfcc', 13)
                n_mels = params.get('n_mels', 128)
                self.register_buffer(
                    '_dct_mfcc',
                    F_afx.dct_matrix(n_mfcc, n_mels),
                    persistent=False,
                )

    # -- introspection ------------------------------------------------------

    def feature_names(self) -> List[str]:
        """Configured feature names, in output order."""
        return list(self._feature_params)

    # -- forward ------------------------------------------------------------

    def _extract_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        params = dict(self._feature_params[name])
        fn = _TORCH_FEATURES[name]
        if name in _SPECTRAL:
            params['window'] = getattr(self, f'_window_{name}')
        if name in ('melspectrogram', 'mfcc'):
            params['mel_fb'] = getattr(self, f'_melfb_{name}')
        if name == 'mfcc':
            params['dct_mat'] = self._dct_mfcc
        if name in ('zcr', 'rms_energy'):
            return fn(x, self.sample_rate, **params)
        return fn(x, self.sample_rate, **params)

    def _aggregate(self, t: torch.Tensor) -> torch.Tensor:
        if self.aggregation == 'none':
            return t
        if self.aggregation == 'mean':
            return t.mean(dim=-1)
        if self.aggregation == 'std':
            return t.std(dim=-1, unbiased=False)
        # mean_std
        return torch.cat(
            [t.mean(dim=-1), t.std(dim=-1, unbiased=False)], dim=-1
        )

    def forward(self, waveform: torch.Tensor):
        """
        Args:
            waveform: (time,) or (batch, time) tensor at ``sample_rate``.

        Returns:
            Dict of tensors (output='dict') or a single (batch, dim) /
            (dim,) tensor (output='vector').
        """
        unbatched = waveform.dim() == 1
        feats: Dict[str, torch.Tensor] = {}
        for name in self._feature_params:
            t = self._extract_one(name, waveform)
            # rms_energy already has a singleton "time" axis of size 1;
            # aggregation over it is a no-op numerically but normalizes
            # the shape with the other features.
            feats[name] = self._aggregate(t)

        if self.output == 'dict':
            return feats

        flat = []
        for name, t in feats.items():
            if unbatched:
                flat.append(t.reshape(-1))
            else:
                flat.append(t.reshape(t.shape[0], -1))
        return torch.cat(flat, dim=-1)

    def extra_repr(self) -> str:
        return (
            f'features={list(self._feature_params)}, '
            f'sample_rate={self.sample_rate}, '
            f'aggregation={self.aggregation!r}, output={self.output!r}'
        )
