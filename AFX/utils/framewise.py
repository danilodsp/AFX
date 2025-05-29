"""
Decorator for frame-wise feature extraction.
"""
from typing import Callable, Dict
import numpy as np
import functools
from AFX.utils.frame_utils import frame_signal


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
                return_metadata: bool = False, **kwargs) -> Dict[str, np.ndarray]:
        if frame_size is None or hop_length is None:
            # Pass hop_length explicitly if provided, even when frame_size is None
            if hop_length is not None:
                return feature_func(signal, sr, *args, hop_length=hop_length, return_metadata=return_metadata, **kwargs)
            else:
                return feature_func(signal, sr, *args, return_metadata=return_metadata, **kwargs)

        # Use our custom framing function
        frames = frame_signal(signal, frame_length=frame_size, hop_length=hop_length)

        results = []
        for i in range(frames.shape[1]):
            res = feature_func(frames[:, i], sr, *args, **kwargs)
            results.append(list(res.values())[0])

        key = list(res.keys())[0]

        # Stack results and ensure we have the right shape
        # The results should be a 1D array with shape (n_frames,)
        stacked = np.concatenate([r.flatten() for r in results])
        result = {key: stacked}

        # Add metadata if requested
        if return_metadata:
            # Calculate frame times: frame_index * hop_length / sr
            n_frames = frames.shape[1]
            times = np.arange(n_frames) * hop_length / sr
            result['metadata'] = {'times': times}
            
        return result
    return wrapper
