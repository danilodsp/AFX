"""
Differentiable, GPU-capable feature extraction in PyTorch.

PyTorch is an optional dependency: ``import afx_audio`` works without it and
does not import this subpackage. Importing ``afx_audio.torch`` or any of its
submodules without PyTorch raises an ImportError with the install command.

Public API:
    - ``afx_audio.torch.functional``: differentiable transforms (stft, mfcc,
      delta, gfcc, chroma_stft, spectral features, ...) mirroring the NumPy
      path.
    - ``AFXFeatureModule``: an ``nn.Module`` for use inside training
      graphs (``.to('cuda')`` supported).
    - ``supported_features()``: names available on the torch path.
"""

try:
    import torch as _torch  # noqa: F401
except ImportError as e:
    # Raised at import, not on attribute access: `import afx_audio.torch.functional`
    # runs this file first, and a lazy guard would let it fail later on a bare `import torch`.
    raise ImportError(
        'afx_audio.torch requires PyTorch, which is not installed. '
        'Install it with `pip install torch` (or `pip install '
        '"afx-audio[torch]"`).'
    ) from e

from afx_audio.torch import functional
from afx_audio.torch.module import AFXFeatureModule, supported_features

__all__ = ['functional', 'AFXFeatureModule', 'supported_features']
