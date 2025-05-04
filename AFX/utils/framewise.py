"""
Decorator for frame-wise feature extraction.
"""
from typing import Callable, Dict
import numpy as np
import functools
import librosa


def framewise_extractor(feature_func: Callable) -> Callable:
    """
    Decorator to apply a feature function frame-wise over an audio signal.
    The decorated function will accept frame_size and hop_length kwargs.
    If frame_size or hop_length are not provided,
    the function is called on the whole signal.
    Returns a dict with the same key as the feature function,
    with values stacked over frames.
    """
    @functools.wraps(feature_func)
    def wrapper(signal: np.ndarray, sr: int, *args,
                frame_size: int = None, hop_length: int = None,
                **kwargs) -> Dict[str, np.ndarray]:
        if frame_size is None or hop_length is None:
            return feature_func(signal, sr, *args, **kwargs)
        frames = librosa.util.frame(
            signal, frame_length=frame_size, hop_length=hop_length
        )
        results = []
        for i in range(frames.shape[1]):
            res = feature_func(frames[:, i], sr, *args, **kwargs)
            results.append(list(res.values())[0])
        key = list(res.keys())[0]
        stacked = np.stack(results, axis=-1)
        return {key: stacked}
    return wrapper
