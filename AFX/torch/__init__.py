"""
AFX.torch — differentiable, GPU-capable feature extraction in PyTorch.

PyTorch is an optional dependency: ``import AFX`` works without it, and
this subpackage raises a clear ImportError only when actually used.

Public API:
    - ``AFX.torch.functional``: differentiable transforms (stft, mfcc,
      melspectrogram, spectral features, ...) mirroring the NumPy path.
    - ``AFXFeatureModule``: an ``nn.Module`` for use inside training
      graphs (``.to('cuda')`` supported).
    - ``supported_features()``: names available on the torch path.
"""

try:
    import torch as _torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

if _TORCH_AVAILABLE:
    from AFX.torch import functional
    from AFX.torch.module import AFXFeatureModule, supported_features

    __all__ = ['functional', 'AFXFeatureModule', 'supported_features']
else:  # pragma: no cover
    __all__ = []

    def __getattr__(name):
        raise ImportError(
            'AFX.torch requires PyTorch, which is not installed. '
            'Install it with `pip install torch` (or `pip install '
            '"AFX[torch]"`).'
        )
