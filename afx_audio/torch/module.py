"""
``AFXFeatureModule``, an ``nn.Module`` wrapper around the differentiable
AFX torch transforms, so feature extraction can sit inside a training
graph (e.g. as the front end of a classifier, or for feature-space losses).
"""
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import nn

from afx_audio.methods.gammatone import gammatone_filterbank as _np_gammatone_filterbank
from afx_audio.torch import functional as F_afx

# Feature name -> functional
_TORCH_FEATURES: dict[str, Callable[..., torch.Tensor]] = {
    'melspectrogram': F_afx.melspectrogram,
    'mfcc': F_afx.mfcc,
    'mfcc_delta': F_afx.mfcc,
    'mfcc_delta_delta': F_afx.mfcc,
    'gfcc': F_afx.gfcc,
    'chroma_stft': F_afx.chroma_stft,
    'spectral_centroid': F_afx.spectral_centroid,
    'spectral_bandwidth': F_afx.spectral_bandwidth,
    'spectral_rolloff': F_afx.spectral_rolloff,
    'spectral_entropy': F_afx.spectral_entropy,
    'spectral_flatness': F_afx.spectral_flatness,
    'spectral_flux': F_afx.spectral_flux,
    'spectral_skewness': F_afx.spectral_skewness,
    'spectral_slope': F_afx.spectral_slope,
    'spectral_kurtosis': F_afx.spectral_kurtosis,
    'spectral_crest': F_afx.spectral_crest,
    'spectral_decrease': F_afx.spectral_decrease,
    'spectral_contrast': F_afx.spectral_contrast,
    'zcr': F_afx.zero_crossing_rate,
    'rms_energy': F_afx.rms_energy,
}

_STFT = {'frame_size', 'hop_length', 'center'}
_MFCC = _STFT | {'n_mfcc', 'n_mels', 'fmin', 'fmax'}
# Params accepted by each functional; anything else raises in __init__ rather
# than silently computing something different from the NumPy path.
_ACCEPTED_PARAMS = {
    'melspectrogram': _STFT | {'n_mels', 'fmin', 'fmax'},
    'mfcc': _MFCC | {'top_db'},
    'mfcc_delta': _MFCC | {'width', 'order'},
    'mfcc_delta_delta': _MFCC | {'width'},
    'gfcc': _STFT | {'n_gfcc', 'n_gammatone', 'fmin', 'fmax'},
    'chroma_stft': _STFT | {'tuning', 'normalize'},
    'spectral_centroid': _STFT,
    'spectral_bandwidth': _STFT,
    'spectral_rolloff': _STFT | {'roll_percent', 'power'},
    'spectral_entropy': _STFT,
    'spectral_flatness': _STFT,
    'spectral_flux': _STFT,
    'spectral_skewness': _STFT,
    'spectral_slope': _STFT,
    'spectral_kurtosis': _STFT | {'fisher'},
    'spectral_crest': _STFT,
    'spectral_decrease': _STFT,
    'spectral_contrast': _STFT | {'n_bands', 'fmin', 'quantile'},
    'zcr': _STFT | {'threshold'},
    'rms_energy': _STFT,
}

_TIME_DOMAIN = {'zcr', 'rms_energy'}
_SPECTRAL = set(_TORCH_FEATURES) - _TIME_DOMAIN
_DELTAS = {'mfcc_delta': 1, 'mfcc_delta_delta': 2}
# Features with a channel axis: (..., n_channels, n_frames)
_TWO_DIM = {'melspectrogram', 'mfcc', 'mfcc_delta', 'mfcc_delta_delta', 'gfcc', 'chroma_stft', 'spectral_contrast'}


def supported_features() -> list[str]:
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
            'mean_std' (both, concatenated; 'summary' is accepted as on
            the NumPy path) over the time axis.
        output: 'dict' returns ``{name: tensor}``; 'vector' flattens and
            concatenates everything into a single (batch, dim) tensor
            (features in insertion order, see ``feature_names()``).

    Example::

        module = AFXFeatureModule(
            features={'mfcc': {'n_mfcc': 13}, 'spectral_centroid': {}},
            sample_rate=16000, aggregation='mean', output='vector',
        ).to('cuda')
        feats = module(waveforms)          # (batch, dim), differentiable
    """

    def __init__(
        self,
        features: dict[str, dict[str, Any]] | list[str],
        sample_rate: int = 22050,
        aggregation: str = 'none',
        output: str = 'dict',
    ):
        super().__init__()
        if isinstance(features, (list, tuple)):
            features = {name: {} for name in features}
        if not features:
            raise ValueError('features must be a non-empty dict or list of names')
        unknown = sorted(set(features) - set(_TORCH_FEATURES))
        if unknown:
            raise ValueError(
                f'Feature(s) not available on the torch path: {unknown}. '
                f'Supported: {supported_features()}. For other features '
                f'use the NumPy path (afx_audio.extract_all_features).'
            )
        if aggregation == 'summary':  # the NumPy name for mean and std
            aggregation = 'mean_std'
        if aggregation not in ('none', 'mean', 'std', 'mean_std'):
            raise ValueError(
                "aggregation must be 'none', 'mean', 'std' or 'mean_std' ('summary')"
            )
        if output not in ('dict', 'vector'):
            raise ValueError("output must be 'dict' or 'vector'")

        self.sample_rate = sample_rate
        self.aggregation = aggregation
        self.output = output
        self._feature_params: dict[str, dict[str, Any]] = {}

        for name, params in features.items():
            params = dict(params or {})
            unsupported = sorted(set(params) - _ACCEPTED_PARAMS[name])
            if unsupported:
                raise ValueError(
                    f'Parameter(s) {unsupported} for {name!r} are not supported on the torch path. '
                    f'Accepted: {sorted(_ACCEPTED_PARAMS[name])}'
                )
            for key in ('frame_size', 'hop_length'):
                if name in _SPECTRAL and key in params and params[key] is None:
                    raise ValueError(f'{name}: whole-clip analysis ({key}=None) is not available on the torch path')
            if name in _DELTAS and params.get('order', _DELTAS[name]) not in (1, 2):
                raise ValueError(f"{name}: order must be 1 or 2, got {params['order']}")
            self._feature_params[name] = params

            # Precompute constants as buffers so they follow .to(device)
            frame_size = params.get('frame_size', 2048)
            if name in _SPECTRAL:
                self.register_buffer(f'_window_{name}', F_afx.hann_window(frame_size), persistent=False)
            if name in ('melspectrogram', 'mfcc', *_DELTAS):
                fb = F_afx.mel_filterbank(params.get('n_mels', 128), frame_size, sr=sample_rate,
                                          fmin=params.get('fmin', 0.0), fmax=params.get('fmax'))
                self.register_buffer(f'_melfb_{name}', fb, persistent=False)
            if name in ('mfcc', *_DELTAS):
                dct = F_afx.dct_matrix(params.get('n_mfcc', 13), params.get('n_mels', 128))
                self.register_buffer(f'_dct_{name}', dct, persistent=False)
            if name == 'gfcc':
                n_gammatone = params.get('n_gammatone', 64)
                fmax = params.get('fmax')
                gammatone = _np_gammatone_filterbank(n_gammatone, frame_size, sample_rate, params.get('fmin', 50.0),
                                                     sample_rate / 2.0 if fmax is None else fmax)
                gammatone_t = torch.as_tensor(np.array(gammatone), dtype=torch.get_default_dtype())
                self.register_buffer('_gammatone_gfcc', gammatone_t, persistent=False)
                self.register_buffer('_dct_gfcc', F_afx.dct_matrix(params.get('n_gfcc', 13), n_gammatone),
                                     persistent=False)
            if name == 'chroma_stft':
                fb = F_afx.chroma_filterbank(sample_rate, frame_size, params.get('tuning', 0.0))
                self.register_buffer('_chroma_chroma_stft', fb, persistent=False)

    # -- introspection ------------------------------------------------------

    def feature_names(self) -> list[str]:
        """Configured feature names, in output order."""
        return list(self._feature_params)

    def get_feature_names_out(self) -> list[str]:
        """
        Column names of the output='vector' tensor, named as on the NumPy path.

        With 'mean_std' these are the 'summary' names (mfcc_mean_0, ...,
        mfcc_std_12); with 'mean' or 'std' the plain names (mfcc_0, ...).
        Not defined for aggregation='none', whose length depends on the
        input duration.
        """
        if self.aggregation == 'none':
            raise ValueError("output names depend on the input length when aggregation='none'")
        buffer = next(self.buffers(), None)
        probe = torch.zeros(int(self.sample_rate), dtype=buffer.dtype if buffer is not None else None,
                            device=buffer.device if buffer is not None else None)
        names: list[str] = []
        with torch.no_grad():
            for name in self._feature_params:
                rows = self._extract_one(name, probe)
                n = rows.shape[0] if name in _TWO_DIM else 1
                stats = ('mean', 'std') if self.aggregation == 'mean_std' else ('',)
                for stat in stats:
                    base = f'{name}_{stat}' if stat else name
                    names.extend([base] if n == 1 else [f'{base}_{i}' for i in range(n)])
        return names

    # -- forward ------------------------------------------------------------

    def _extract_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        params = dict(self._feature_params[name])
        if name in _SPECTRAL:
            params['window'] = getattr(self, f'_window_{name}')
        if name in ('melspectrogram', 'mfcc', *_DELTAS):
            params['mel_fb'] = getattr(self, f'_melfb_{name}')
        if name in ('mfcc', *_DELTAS, 'gfcc'):
            params['dct_mat'] = getattr(self, f'_dct_{name}')
        if name == 'gfcc':
            params['gammatone_fb'] = self._gammatone_gfcc
        if name == 'chroma_stft':
            params['chroma_fb'] = self._chroma_chroma_stft
            params.pop('tuning', None)
        if name in _DELTAS:
            width = params.pop('width', 9)
            order = params.pop('order', _DELTAS[name])
            return F_afx.delta(F_afx.mfcc(x, self.sample_rate, **params), width=width, order=order)
        return _TORCH_FEATURES[name](x, self.sample_rate, **params)

    @staticmethod
    def _nan_mean_std(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Frames where a feature is undefined (NaN, e.g. spectral kurtosis of
        # silence) are skipped, as in the NumPy aggregation. torch.where keeps
        # the NaN entries out of the gradient.
        valid = ~torch.isnan(t)
        count = valid.sum(dim=-1)
        values = torch.where(valid, t, torch.zeros_like(t))
        safe = count.clamp(min=1).to(t.dtype)
        mean = values.sum(dim=-1) / safe
        dev = torch.where(valid, t - mean.unsqueeze(-1), torch.zeros_like(t))
        std = F_afx._safe_sqrt((dev ** 2).sum(dim=-1) / safe)
        nan = torch.full_like(mean, float('nan'))
        empty = count == 0
        return torch.where(empty, nan, mean), torch.where(empty, nan, std)

    def _aggregate(self, t: torch.Tensor, has_channels: bool) -> torch.Tensor:
        if self.aggregation == 'none':
            return t
        mean, std = self._nan_mean_std(t)
        if self.aggregation == 'mean':
            return mean
        if self.aggregation == 'std':
            return std
        if has_channels:
            # (..., n, T) -> (..., 2n): all means, then all stds, matching
            # the NumPy 'summary' aggregation order.
            return torch.cat([mean, std], dim=-1)
        # (..., T) -> (..., 2). Concatenating along dim=-1 here would merge
        # the batch axis and fail on unbatched input.
        return torch.stack([mean, std], dim=-1)

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor] | torch.Tensor:
        """
        Features of a (time,) or (batch, time) waveform at ``sample_rate``.

        A dict of tensors for output='dict', otherwise one (batch, dim) or
        (dim,) tensor.
        """
        if not isinstance(waveform, torch.Tensor):
            raise TypeError(f'expected a torch.Tensor waveform, got {type(waveform).__name__}; use torch.as_tensor(x)')
        unbatched = waveform.dim() == 1
        feats: dict[str, torch.Tensor] = {}
        for name in self._feature_params:
            t = self._extract_one(name, waveform)
            # Whole-clip rms_energy has a time axis of length 1, so it
            # aggregates to the same shape as the framed features.
            feats[name] = self._aggregate(t, name in _TWO_DIM)

        if self.output == 'dict':
            return feats

        flat = []
        for t in feats.values():
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
