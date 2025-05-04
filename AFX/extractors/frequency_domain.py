import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor

__all__ = [
    'extract_spectral_centroid',
    'extract_spectral_bandwidth',
    'extract_spectral_rolloff',
]


@framewise_extractor
def extract_spectral_centroid(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral centroid of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_centroid' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: Hz
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    centroid = librosa.feature.spectral_centroid(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]
    result = {'spectral_centroid': centroid}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(centroid)), sr=sr, hop_length=hop_length,
            n_fft=frame_size
        )
        return {'spectral_centroid': centroid, 'metadata': {'times': times}}
    return result


@framewise_extractor
def extract_spectral_bandwidth(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral bandwidth of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_bandwidth' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: Hz
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    bandwidth = librosa.feature.spectral_bandwidth(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )[0]
    result = {'spectral_bandwidth': bandwidth}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(bandwidth)), sr=sr, hop_length=hop_length,
            n_fft=frame_size
        )
        return {'spectral_bandwidth': bandwidth, 'metadata': {'times': times}}
    return result


@framewise_extractor
def extract_spectral_rolloff(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    roll_percent: float = 0.85,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral rolloff of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        roll_percent: Roll-off percentage (default 0.85)
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_rolloff' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: Hz
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length,
        roll_percent=roll_percent
    )[0]
    result = {'spectral_rolloff': rolloff}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(rolloff)), sr=sr, hop_length=hop_length,
            n_fft=frame_size
        )
        return {'spectral_rolloff': rolloff, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_spectral_contrast(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral contrast of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        n_bands: Number of frequency bands
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_contrast' key and np.ndarray of values
    Metadata:
        shape: (n_bands, n_frames)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    contrast = librosa.feature.spectral_contrast(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length, n_bands=n_bands
    )
    result = {'spectral_contrast': contrast}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(contrast.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'spectral_contrast': contrast, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_spectral_entropy(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bins: int = 128,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral entropy of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        n_bins: Number of frequency bins
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_entropy' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: float
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    S = np.abs(librosa.stft(signal, n_fft=frame_size, hop_length=hop_length))
    ps = S / (np.sum(S, axis=0, keepdims=True) + 1e-10)
    entropy = -np.sum(ps * np.log2(ps + 1e-10), axis=0)
    result = {'spectral_entropy': entropy}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(entropy.shape[0]), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'spectral_entropy': entropy, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_spectral_flatness(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral flatness of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_flatness' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: float
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    flatness = librosa.feature.spectral_flatness(
        y=signal, n_fft=frame_size, hop_length=hop_length
    )[0]
    result = {'spectral_flatness': flatness}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(flatness)), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'spectral_flatness': flatness, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_spectral_flux(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral flux of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_flux' key and np.ndarray of values
    Metadata:
        shape: (n_frames,)
        units: float
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    S = np.abs(librosa.stft(signal, n_fft=frame_size, hop_length=hop_length))
    flux = np.sqrt(np.sum(np.diff(S, axis=1) ** 2, axis=0))
    result = {'spectral_flux': flux}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(len(flux)), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'spectral_flux': flux, 'metadata': {'times': times}}
    return result
