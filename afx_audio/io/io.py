"""
Audio file loading.
"""
import math
import os
from typing import Any

import numpy as np
import soundfile as sf


def load_audio(
    path: str | os.PathLike | Any,
    sr: int | None = 22050,
    offset: float = 0.0,
    duration: float | None = None,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono float32 signal.

    Channels are averaged. With ``sr`` set the audio is resampled with a
    polyphase filter (scipy.signal.resample_poly); ``sr=None`` keeps the
    file's rate. ``offset`` and ``duration`` (seconds, in the file's
    timeline) read only part of the file. ``path`` may also be a file
    object, as accepted by soundfile.

    Returns:
        (signal, sample_rate)

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: no samples in the requested range, or a negative
            offset or non-positive duration.
        RuntimeError: soundfile cannot read the file.
    """
    if offset < 0:
        raise ValueError(f'offset must be >= 0, got {offset}')
    if duration is not None and duration <= 0:
        raise ValueError(f'duration must be positive, got {duration}')
    try:
        with sf.SoundFile(path) as f:
            native_sr = f.samplerate
            start = int(round(offset * native_sr))
            if start:
                f.seek(min(start, f.frames))
            frames = -1 if duration is None else int(round(duration * native_sr))
            signal = f.read(frames=frames, dtype='float32', always_2d=True)
    except FileNotFoundError:
        raise
    except Exception as e:
        if isinstance(path, (str, os.PathLike)) and not os.path.exists(path):
            raise FileNotFoundError(f'Audio file not found: {path}') from e
        raise RuntimeError(f'Error loading audio file {path}: {e}') from e

    if signal.shape[0] == 0:
        raise ValueError(f'No audio samples in {path} (offset {offset} s, duration {duration})')

    signal = signal.mean(axis=1) if signal.shape[1] > 1 else signal[:, 0]

    if sr is None or sr == native_sr:
        return signal.astype(np.float32), int(native_sr)

    from scipy.signal import resample_poly

    g = math.gcd(int(sr), int(native_sr))
    signal = resample_poly(signal, int(sr) // g, int(native_sr) // g)
    return signal.astype(np.float32), int(sr)
