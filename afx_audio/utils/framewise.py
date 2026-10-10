"""
Decorator that turns a whole-signal feature function into a frame-wise one.
"""
import functools
import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from afx_audio.utils.frame_utils import analysis_frames, frame_times


def framewise_extractor(feature_func: Callable) -> Callable:
    """
    Run ``feature_func(frame, sr, **params)`` on every frame and stack the results.

    The wrapped function gains keyword-only ``frame_size``, ``hop_length``,
    ``center`` and ``return_metadata`` arguments; framing follows
    afx_audio.utils.frame_utils.analysis_frames (whole signal when either is None
    or the clip is shorter than a frame). ``feature_func`` must return a
    dict with exactly one key; scalar results stack to (n_frames,), vector
    results to (n_dims, n_frames).

    This calls Python once per frame. The built-in extractors are
    vectorized; use it for custom features where that does not matter.
    """
    @functools.wraps(feature_func)
    def wrapper(signal: np.ndarray, sr: int | None = None, *, frame_size: int | None = None,
                hop_length: int | None = None, center: bool = True, return_metadata: bool = False,
                **params) -> dict[str, Any]:
        frames = analysis_frames(signal, frame_size, hop_length, center)
        key = None
        results = []
        for frame in frames:
            res = feature_func(np.array(frame), sr, **params)
            if key is None:
                keys = [k for k in res if k != 'metadata']
                if len(keys) != 1:
                    raise ValueError(
                        f'{feature_func.__name__} returned {len(keys)} feature '
                        f'keys {keys}; framewise extraction supports exactly one.'
                    )
                key = keys[0]
            results.append(np.asarray(res[key]))

        first_shape = results[0].shape
        if first_shape in ((), (1,)):
            stacked = np.array(results).reshape(-1)
        else:
            if first_shape[-1] == 1:
                results = [np.squeeze(r, axis=-1) for r in results]
            stacked = np.stack(results, axis=-1)
        out: dict[str, Any] = {str(key): stacked}
        if return_metadata:
            if sr is None:
                raise ValueError('sr is required when return_metadata=True')
            centered = center and frame_size is not None and hop_length is not None
            out['metadata'] = {'times': frame_times(len(frames), sr, hop_length or 0, frames.shape[1], centered)}
        return out

    inner = [p for p in inspect.signature(feature_func).parameters.values()
             if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)]
    framing = [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY, default=default)
               for name, default in (('frame_size', None), ('hop_length', None), ('center', True),
                                         ('return_metadata', False))]
    inner = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) if i >= 2 else p for i, p in enumerate(inner)]
    wrapper.__signature__ = inspect.Signature(inner[:2] + framing + inner[2:])  # type: ignore[attr-defined]
    return wrapper
