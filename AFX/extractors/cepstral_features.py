"""
Cepstral feature extractors for audio signals (MFCC, delta, etc.).
"""
import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor

__all__ = [
    'extract_mfcc',
    'extract_mfcc_delta',
    'extract_mfcc_delta_delta',
    'extract_chroma_cqt',
    'extract_chroma_stft',
    'extract_cqt',
    'extract_melspectrogram',
]

@framewise_extractor
def extract_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute MFCCs from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        n_mfcc: Number of MFCCs
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'mfcc' key and np.ndarray of shape (n_mfcc, n_frames)
    Metadata:
        shape: (n_mfcc, n_frames)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=frame_size, hop_length=hop_length
    )
    result = {'mfcc': mfcc}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(mfcc.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'mfcc': mfcc, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_mfcc_delta(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    order: int = 1,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute delta or delta-delta MFCCs from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        n_mfcc: Number of MFCCs
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        order: Delta order (1=delta, 2=delta-delta)
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'mfcc_delta' key and np.ndarray of shape (n_mfcc, n_frames)
    Metadata:
        shape: (n_mfcc, n_frames)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=frame_size,
        hop_length=hop_length
    )
    n_frames = mfcc.shape[1]
    # Ensure width is odd, >=3, and does not exceed n_frames
    width = min(9, n_frames)
    if width < 3:
        delta = np.zeros_like(mfcc)
    else:
        if width % 2 == 0:
            width -= 1
        delta = librosa.feature.delta(
            mfcc, order=order, width=width
        )
    result = {'mfcc_delta': delta}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(delta.shape[1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=frame_size
        )
        return {'mfcc_delta': delta, 'metadata': {'times': times}}
    return result


@framewise_extractor
def extract_mfcc_delta_delta(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute delta-delta (acceleration) MFCCs from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        n_mfcc: Number of MFCCs
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'mfcc_delta_delta' key and np.ndarray of shape
        (n_mfcc, n_frames)
    Metadata:
        shape: (n_mfcc, n_frames)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    mfcc = librosa.feature.mfcc(
        y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=frame_size,
        hop_length=hop_length
    )
    n_frames = mfcc.shape[1]
    width = min(9, n_frames)
    if width < 3:
        delta2 = np.zeros_like(mfcc)
    else:
        if width % 2 == 0:
            width -= 1
        delta2 = librosa.feature.delta(mfcc, order=2, width=width)
    result = {'mfcc_delta_delta': delta2}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(delta2.shape[1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=frame_size
        )
        return {'mfcc_delta_delta': delta2, 'metadata': {'times': times}}
    return result



@framewise_extractor

def extract_chroma_cqt(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute chroma CQT features from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for CQT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'chroma_cqt' key and np.ndarray of shape (12, n_frames)
    Metadata:
        shape: (12, n_frames)
        units: amplitude
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    chroma = librosa.feature.chroma_cqt(
        y=signal, sr=sr, hop_length=hop_length
    )
    result = {'chroma_cqt': chroma}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=frame_size
        )
        return {'chroma_cqt': chroma, 'metadata': {'times': times}}
    return result


@framewise_extractor

def extract_chroma_stft(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute chroma STFT features from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'chroma_stft' key and np.ndarray of shape (12, n_frames)
    Metadata:
        shape: (12, n_frames)
        units: amplitude
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    chroma = librosa.feature.chroma_stft(
        y=signal, sr=sr, n_fft=frame_size, hop_length=hop_length
    )
    result = {'chroma_stft': chroma}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=sr,
            hop_length=hop_length,
            n_fft=frame_size
        )
        return {'chroma_stft': chroma, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_cqt(
    signal: np.ndarray,
    sr: int,
    hop_length: int = 512,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the Constant-Q Transform (CQT) of an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        hop_length: Hop length between frames
        n_bins: Number of frequency bins
        bins_per_octave: Bins per octave
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'cqt' key and np.ndarray of shape (n_bins, n_frames)
    Metadata:
        shape: (n_bins, n_frames)
        units: amplitude
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    cqt = np.abs(librosa.cqt(
        y=signal, sr=sr, hop_length=hop_length, n_bins=n_bins, bins_per_octave=bins_per_octave
    ))
    result = {'cqt': cqt}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(cqt.shape[1]), sr=sr, hop_length=hop_length
        )
        return {'cqt': cqt, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_melspectrogram(
    signal: np.ndarray,
    sr: int,
    n_mels: int = 128,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute Mel-scaled spectrogram from an audio signal.
    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        n_mels: Number of Mel bands
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'melspectrogram' key and np.ndarray of shape (n_mels, n_frames)
    Metadata:
        shape: (n_mels, n_frames)
        units: power
        times: np.ndarray, shape (n_frames,) if return_metadata is True
    """
    import librosa
    mel = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_mels=n_mels, n_fft=frame_size, hop_length=hop_length
    )
    result = {'melspectrogram': mel}
    if return_metadata:
        times = librosa.frames_to_time(
            np.arange(mel.shape[1]), sr=sr, hop_length=hop_length, n_fft=frame_size
        )
        return {'melspectrogram': mel, 'metadata': {'times': times}}
    return result
