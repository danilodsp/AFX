"""
Cepstral feature extractors for audio signals (MFCC, delta, etc.).
"""
import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor
from AFX.methods.cqt import cqt_approximation
from AFX.methods.stft import stft, power_spectrogram, magnitude_spectrogram
from AFX.methods.mel import mel_filterbank, apply_mel_filterbank, log_mel_spectrogram
from AFX.methods.dct import extract_mfcc_coefficients
from AFX.methods.delta import compute_delta
from AFX.methods.chroma import extract_chroma_from_cqt, chroma_from_stft


from AFX.methods.gammatone import gammatone_filterbank, apply_gammatone_filterbank

__all__ = [
    'extract_mfcc',
    'extract_mfcc_delta',
    'extract_mfcc_delta_delta',
    'extract_chroma_cqt',
    'extract_chroma_stft',
    'extract_cqt',
    'extract_melspectrogram',
    'extract_gfcc',
]

@framewise_extractor
def extract_mfcc(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float = None,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute MFCCs from an audio signal using NumPy/SciPy implementation.

    The MFCC computation pipeline:
    1. Apply Short-Time Fourier Transform (STFT) with Hann window
    2. Compute power spectrogram from STFT magnitude  
    3. Map power spectrum through triangular mel-scale filterbank
    4. Take logarithm of mel energies (with clamping to avoid log(0))
    5. Apply Discrete Cosine Transform type-II to decorrelate features

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients to extract (default: 13)
        frame_size: Frame size for STFT (n_fft) (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        n_mels: Number of mel filterbank bands (default: 128)
        fmin: Minimum frequency for mel filterbank in Hz (default: 0.0)
        fmax: Maximum frequency for mel filterbank in Hz (default: sr/2)
        center: If True, pad the signal so that frames are centered
            (matching ``librosa`` behavior). When ``preserve_shape`` is used,
            setting ``center=False`` yields the standard ``(n_features, n_frames)``
            shape with no extra padding frames.
        return_metadata: If True, include frame times in metadata

    Returns:
        Dict with 'mfcc' key and np.ndarray of shape (n_mfcc, n_frames)

    Metadata:
        shape: (n_mfcc, n_frames)
        units: Dimensionless (cepstral coefficients)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses Hann windowing and optional center padding for STFT
        - Mel filterbank uses Slaney normalization 
        - DCT-II with orthogonal normalization extracts cepstral coefficients
        - Frame times are computed as frame_index * hop_length / sr
    """
    # Set default fmax
    if fmax is None:
        fmax = sr / 2.0

    # 1. Compute STFT with Hann window
    stft_matrix = stft(
        signal,
        frame_size=frame_size,
        hop_length=hop_length,
        window='hann',
        center=center,
    )

    # 2. Compute power spectrogram
    power_spec = power_spectrogram(stft_matrix)

    # 3. Create and apply mel filterbank
    filterbank = mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm='slaney')
    mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # 4. Convert to log scale
    log_mel = log_mel_spectrogram(mel_spec, amin=1e-10, top_db=80.0)

    # 5. Extract MFCC coefficients using DCT-II
    mfcc = extract_mfcc_coefficients(log_mel, n_mfcc, norm='ortho')

    result = {'mfcc': mfcc}
    if return_metadata:
        # Calculate frame times manually using hop_length / sr
        n_frames = mfcc.shape[1]
        frame_offset = frame_size // 2 if center else 0
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
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
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute delta or delta-delta MFCCs from an audio signal using internal implementation.

    This function computes MFCC features using the internal implementation and then applies
    delta (first derivative) or delta-delta (second derivative) computation using a 
    finite difference kernel-based method.

    The delta computation uses the formula:
    delta[t] = sum_{n=-N}^{N} w[n] * x[t+n]

    where w is the delta kernel:
    w[n] = n / (2 * sum(i^2 for i in range(1, (width+1)//2 + 1)))

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients (default: 13)
        frame_size: Frame size for STFT (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        order: Delta order (1=delta, 2=delta-delta) (default: 1)
        center: If True, pad the signal so that frames are centered
            before computing the STFT.
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional arguments passed to internal MFCC computation

    Returns:
        Dict with 'mfcc_delta' key and np.ndarray of shape (n_mfcc, n_frames)

    Metadata:
        shape: (n_mfcc, n_frames)
        units: Dimensionless (delta cepstral coefficients)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses internal MFCC computation
        - Delta computation uses finite difference with edge padding
        - Width of delta kernel is automatically set to min(9, n_frames) 
        - Returns zeros if insufficient frames (< 3) for delta computation
    """
    # Compute MFCCs directly (not through the decorator)
    # This avoids the framewise reshaping that would happen with extract_mfcc

    # Set defaults
    n_mels = kwargs.get('n_mels', 128)
    fmin = kwargs.get('fmin', 0.0)
    fmax = kwargs.get('fmax', sr / 2.0)

    # 1. Compute STFT with Hann window and center padding
    stft_matrix = stft(
        signal,
        frame_size=frame_size,
        hop_length=hop_length,
        window='hann',
        center=center,
    )

    # 2. Compute power spectrogram
    power_spec = power_spectrogram(stft_matrix)

    # 3. Create and apply mel filterbank
    filterbank = mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm='slaney')
    mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # 4. Convert to log scale
    log_mel = log_mel_spectrogram(mel_spec, amin=1e-10, top_db=80.0)

    # 5. Extract MFCC coefficients using DCT-II
    mfcc = extract_mfcc_coefficients(log_mel, n_mfcc, norm='ortho')

    n_frames = mfcc.shape[1]

    # Ensure width is odd, >=3, and does not exceed n_frames
    width = min(9, n_frames)
    if width < 3:
        delta = np.zeros_like(mfcc)
    else:
        if width % 2 == 0:
            width -= 1

        # Compute delta using internal implementation
        delta = compute_delta(mfcc, width=width, order=order)

    result = {'mfcc_delta': delta}

    if return_metadata:
        # Calculate frame times manually using hop_length / sr
        n_frames = delta.shape[1]
        frame_offset = frame_size // 2 if center else 0
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
        return {'mfcc_delta': delta, 'metadata': {'times': times}}

    return result


@framewise_extractor
def extract_mfcc_delta_delta(
    signal: np.ndarray,
    sr: int,
    n_mfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute delta-delta (acceleration) MFCCs from an audio signal using internal implementation.

    This function computes MFCC features using the internal implementation and then applies
    delta-delta (second derivative) computation using a finite difference kernel-based method.

    The acceleration computation uses the formula:
    delta_delta[t] = sum_{n=-N}^{N} w[n] * delta[t+n]

    where delta is the first derivative and w is the delta kernel:
    w[n] = n / (2 * sum(i^2 for i in range(1, (width+1)//2 + 1)))

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients (default: 13)
        frame_size: Frame size for STFT (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        center: If True, pad the signal so that frames are centered
            before computing the STFT.
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional arguments passed to internal MFCC computation

    Returns:
        Dict with 'mfcc_delta_delta' key and np.ndarray of shape (n_mfcc, n_frames)

    Metadata:
        shape: (n_mfcc, n_frames)
        units: Dimensionless (delta-delta cepstral coefficients)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses internal MFCC computation
        - Delta-delta computation uses finite difference with edge padding
        - Width of delta kernel is automatically set to min(9, n_frames)
        - Returns zeros if insufficient frames (< 3) for delta computation
        - Computes second-order derivatives (acceleration) of MFCC features
    """
    # Compute MFCCs directly using internal implementation
    # This avoids the framewise reshaping that would happen with extract_mfcc

    # Set defaults
    n_mels = kwargs.get('n_mels', 128)
    fmin = kwargs.get('fmin', 0.0)
    fmax = kwargs.get('fmax', sr / 2.0)

    # 1. Compute STFT with Hann window
    stft_matrix = stft(
        signal,
        frame_size=frame_size,
        hop_length=hop_length,
        window='hann',
        center=center,
    )

    # 2. Compute power spectrogram
    power_spec = power_spectrogram(stft_matrix)

    # 3. Create and apply mel filterbank
    filterbank = mel_filterbank(n_mels, frame_size, sr, fmin, fmax, norm='slaney')
    mel_spec = apply_mel_filterbank(power_spec, filterbank)

    # 4. Convert to log scale
    log_mel = log_mel_spectrogram(mel_spec, amin=1e-10, top_db=80.0)

    # 5. Extract MFCC coefficients using DCT-II
    mfcc = extract_mfcc_coefficients(log_mel, n_mfcc, norm='ortho')

    n_frames = mfcc.shape[1]

    # Ensure width is odd, >=3, and does not exceed n_frames
    width = min(9, n_frames)
    if width < 3:
        delta_delta = np.zeros_like(mfcc)
    else:
        if width % 2 == 0:
            width -= 1

        # Compute delta-delta using internal implementation with order=2
        delta_delta = compute_delta(mfcc, width=width, order=2)

    result = {'mfcc_delta_delta': delta_delta}

    if return_metadata:
        # Calculate frame times manually using hop_length / sr
        # Account for center padding offset
        n_frames = delta_delta.shape[1]
        frame_offset = frame_size // 2 if center else 0
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
        return {'mfcc_delta_delta': delta_delta, 'metadata': {'times': times}}

    return result

@framewise_extractor
def extract_gfcc(
    signal: np.ndarray,
    sr: int,
    n_gfcc: int = 13,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_gammatone: int = 64,
    fmin: float = 50.0,
    fmax: float = None,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute Gammatone Frequency Cepstral Coefficients (GFCC) from an audio signal.

    The GFCC computation pipeline:
    1. Apply Short-Time Fourier Transform (STFT) with Hann window
    2. Compute power spectrogram from STFT magnitude
    3. Map power spectrum through gammatone filterbank
    4. Take logarithm of gammatone energies (with clamping to avoid log(0))
    5. Apply Discrete Cosine Transform type-II to decorrelate features

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        n_gfcc: Number of GFCC coefficients to extract (default: 13)
        frame_size: Frame size for STFT (n_fft) (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        n_gammatone: Number of gammatone filterbank bands (default: 64)
        fmin: Minimum frequency for gammatone filterbank in Hz (default: 50.0)
        fmax: Maximum frequency for gammatone filterbank in Hz (default: sr/2)
        center: If True, pad the signal so that frames are centered
            before computing the STFT.
        return_metadata: If True, include frame times in metadata

    Returns:
        Dict with 'gfcc' key and np.ndarray of shape (n_gfcc, n_frames)

    Metadata:
        shape: (n_gfcc, n_frames)
        units: Dimensionless (cepstral coefficients)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses Hann windowing and optional center padding for STFT
        - Gammatone filterbank is ERB-spaced
        - DCT-II with orthogonal normalization extracts cepstral coefficients
        - Frame times are computed as frame_index * hop_length / sr
    """
    if fmax is None:
        fmax = sr / 2.0

    # 1. Compute STFT with Hann window
    stft_matrix = stft(
        signal,
        frame_size=frame_size,
        hop_length=hop_length,
        window='hann',
        center=center,
    )

    # 2. Compute power spectrogram
    power_spec = power_spectrogram(stft_matrix)

    # 3. Create and apply gammatone filterbank
    filterbank = gammatone_filterbank(n_gammatone, frame_size, sr, fmin, fmax)
    gammatone_spec = apply_gammatone_filterbank(power_spec, filterbank)

    # 4. Convert to log scale
    log_gammatone = np.log(np.maximum(gammatone_spec, 1e-10))

    # 5. Extract GFCC coefficients using DCT-II
    from AFX.methods.dct import extract_mfcc_coefficients
    gfcc = extract_mfcc_coefficients(log_gammatone, n_gfcc, norm='ortho')

    result = {'gfcc': gfcc}
    if return_metadata:
        n_frames = gfcc.shape[1]
        frame_offset = frame_size // 2 if center else 0
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
        return {'gfcc': gfcc, 'metadata': {'times': times}}
    return result

@framewise_extractor
def extract_chroma_cqt(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    center: bool = False,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute chroma CQT features from an audio signal using internal implementation.

    This method computes a chromagram from a Constant-Q Transform (CQT) approximation,
    capturing harmonic content and mapping it to 12 pitch classes. The implementation:

    1. Computes a CQT approximation using log-spaced frequency bins and STFT
    2. Maps frequency bins to chroma bins (12 pitch classes, modulo 12)
    3. Aggregates energy into each chroma bin across all octaves

    Mathematical formulation:
    - CQT frequencies: f[k] = fmin * 2^(k/bins_per_octave)
    - MIDI note mapping: midi = 69 + 12 * log2(freq / 440)
    - Chroma bin: chroma_bin = midi % 12
    - Aggregation: chroma[c] = sum(cqt_mag[k] for all k where bin[k] maps to c)

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        frame_size: Frame size for STFT used in CQT approximation (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        center: If True, pad the signal so that frames are centered in the STFT
            used for the CQT approximation.
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional parameters (n_bins, bins_per_octave, fmin)

    Returns:
        Dict with 'chroma_cqt' key and np.ndarray of shape (12, n_frames)

    Metadata:
        shape: (12, n_frames) where each row is a pitch class [C, C#, D, D#, E, F, F#, G, G#, A, A#, B]
        units: Summed magnitude (energy) for each pitch class
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses geometric frequency spacing to approximate CQT behavior
        - Default range covers ~7 octaves from C1 (32.7 Hz) to C8 (4186 Hz)
        - Higher frame_size provides better frequency resolution for low frequencies
        - Chroma bins activate based on harmonic content presence in corresponding pitch classes
    """
    n_bins = kwargs.get('n_bins', 84)  # 7 octaves * 12 bins per octave
    bins_per_octave = kwargs.get('bins_per_octave', 12)
    fmin = kwargs.get('fmin', 32.70319566257483)  # C1 in Hz

    # Compute chroma features using internal CQT approximation
    chroma = extract_chroma_from_cqt(
        signal,
        sr=sr,
        hop_length=hop_length,
        n_bins=n_bins,
        bins_per_octave=bins_per_octave,
        fmin=fmin,
        center=center,
    )

    result = {'chroma_cqt': chroma}

    if return_metadata:
        # Calculate frame times manually. The internal STFT uses a frame size of
        # 4096 samples for the CQT approximation.
        n_frames = chroma.shape[1]
        frame_offset = 4096 // 2 if center else 0
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
        return {'chroma_cqt': chroma, 'metadata': {'times': times}}

    return result


def extract_chroma_stft(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute chroma STFT features from an audio signal using internal implementation.

    Extracts a chromagram using Short-Time Fourier Transform (STFT) and projects
    the frequency content onto 12 chroma bins (pitch classes).

    The process involves:
    1. Compute magnitude spectrogram from STFT using high-resolution FFT
    2. Map frequency bins to chroma bins using: chroma_bin = round(12 * log2(f / f_ref)) % 12
    3. Aggregate magnitudes into chroma bins across all octaves for each frame
    4. Optional L1 normalization per frame (each chroma vector sums to 1)

    Mathematical Details:
    - Frequency bins: f = bin_index * sr / n_fft
    - Chroma mapping: chroma_bin = round(12 * log2(f / f_ref)) % 12
    - Reference frequency: f_ref = 261.626 Hz (C4)
    - Aggregation: sum all frequency bins mapping to same chroma bin
    - Normalization: chroma[i, :] = chroma[i, :] / sum(chroma[:, :], axis=0)

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        frame_size: Frame size for STFT (n_fft) (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional arguments (f_ref for reference frequency, normalize for L1 norm)

    Returns:
        Dict with 'chroma_stft' key and np.ndarray of shape (12, n_frames)

    Metadata:
        shape: (12, n_frames)
        units: Normalized amplitude (if normalize=True)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses Hann windowing and center padding for STFT
        - Reference frequency f_ref defaults to 261.626 Hz (C4)
        - DC bin (f=0) is ignored to avoid log(0) in chroma mapping
        - Chroma bin mapping: 0=C, 1=C#, 2=D, 3=D#, 4=E, 5=F, 6=F#, 7=G, 8=G#, 9=A, 10=A#, 11=B
        - Frame times computed as: (frame_index * hop_length + frame_size//2) / sr
    """
    # Extract optional parameters
    f_ref = kwargs.get('f_ref', 261.626)  # C4 frequency
    normalize = kwargs.get('normalize', True)

    # 1. Compute STFT with Hann window and center padding
    stft_matrix = stft(signal, frame_size=frame_size, hop_length=hop_length, 
                       window='hann', center=True)

    # 2. Compute magnitude spectrogram
    mag_spec = magnitude_spectrogram(stft_matrix)

    # 3. Extract chroma features from magnitude spectrogram
    chroma = chroma_from_stft(mag_spec, sr=sr, n_fft=frame_size, 
                              f_ref=f_ref, normalize=normalize)

    result = {'chroma_stft': chroma}
    if return_metadata:
        # Calculate frame times manually using hop_length and sr
        # Account for center padding offset
        n_frames = chroma.shape[1]
        frame_offset = frame_size // 2  # Center padding offset
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
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
    Compute the Constant-Q Transform (CQT) of an audio signal using internal implementation.

    This function approximates the Constant-Q Transform by:
    1. Computing STFT with high frequency resolution (n_fft=2048)
    2. Creating logarithmically-spaced frequency bins (geometric series)
    3. Mapping STFT bins to CQT bins using linear interpolation

    The CQT uses filters whose bandwidths increase exponentially with frequency,
    providing better resolution at low frequencies. This implementation uses
    log-frequency projection of the STFT to approximate this behavior.

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        hop_length: Hop length between frames in samples
        n_bins: Number of frequency bins
        bins_per_octave: Bins per octave (typically 12)
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional parameters (fmin, n_fft)

    Returns:
        Dict with 'cqt' key and np.ndarray of shape (n_bins, n_frames)

    Metadata:
        shape: (n_bins, n_frames)
        units: magnitude (approximated from STFT)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses geometric frequency spacing: f[k] = fmin * 2^(k/bins_per_octave)
        - Default minimum frequency is C1 (≈32.7 Hz)
        - Higher n_fft provides better frequency resolution for low frequencies
        - This is an approximation of the true CQT using log-frequency filterbanks
    """
    # Extract additional parameters with defaults
    fmin = kwargs.get('fmin', None)  # C1 will be used as default in cqt_approximation
    n_fft = kwargs.get('n_fft', 2048)  # High resolution for better frequency precision

    # Compute CQT approximation using internal implementation
    cqt_magnitude, _ = cqt_approximation(
        signal, sr=sr, hop_length=hop_length, n_bins=n_bins, 
        bins_per_octave=bins_per_octave, fmin=fmin, frame_size=n_fft
    )

    result = {'cqt': cqt_magnitude}

    if return_metadata:
        # Calculate frame times manually: frame_index * hop_length / sr
        n_frames = cqt_magnitude.shape[1]
        times = np.arange(n_frames) * hop_length / sr
        return {'cqt': cqt_magnitude, 'metadata': {'times': times}}

    return result

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
    Compute Mel-scaled spectrogram from an audio signal using internal implementation.

    This method computes a time-frequency representation of the signal on the Mel scale,
    using triangular filterbanks applied to the power spectrum. The implementation:

    1. Computes power spectrogram via STFT with Hann windowing and center padding
    2. Creates Mel filterbank matrix with triangular filters spaced in Mel scale
    3. Applies filterbank to project spectral energy onto Mel bands
    4. Returns result with shape (n_mels, n_frames) containing non-negative power values

    Mathematical formulation:
    - STFT: X[f,t] = sum(x[n] * w[n-t*H] * exp(-j*2*pi*f*n/N))
    - Power spectrum: P[f,t] = |X[f,t]|^2  
    - Mel filterbank: M[m,f] with triangular filters on mel-frequency scale
    - Mel spectrogram: S[m,t] = sum(M[m,f] * P[f,t]) over frequency f

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate in Hz
        n_mels: Number of Mel bands (default: 128)
        frame_size: Frame size for STFT (n_fft) (default: 2048)
        hop_length: Hop length between frames in samples (default: 512)
        return_metadata: If True, include frame times in metadata
        **kwargs: Additional parameters (fmin, fmax for frequency range)

    Returns:
        Dict with 'melspectrogram' key and np.ndarray of shape (n_mels, n_frames)

    Metadata:
        shape: (n_mels, n_frames)
        units: power (spectral energy in each mel band)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Notes:
        - Uses Hann windowing and center padding for STFT
        - Mel filterbank uses Slaney normalization and scale (htk=False)
        - Filter shapes are triangular with linear interpolation between mel frequencies
        - Frame times computed as: (frame_index * hop_length + frame_size//2) / sr
        - Frequency range spans from fmin (default: 0 Hz) to fmax (default: sr/2)
        - All output values are non-negative (power domain)
    """
    # Extract optional frequency range parameters
    fmin = kwargs.get('fmin', 0.0)
    fmax = kwargs.get('fmax', sr / 2.0)

    # 1. Compute STFT with Hann window and center padding
    stft_matrix = stft(signal, frame_size=frame_size, hop_length=hop_length, 
                       window='hann', center=True)

    # 2. Compute power spectrogram (squared magnitude)
    power_spec = power_spectrogram(stft_matrix)

    # 3. Create mel filterbank with Slaney normalization
    filterbank = mel_filterbank(n_mels, frame_size, sr, fmin=fmin, fmax=fmax, 
                               norm='slaney', htk=False)

    # 4. Apply mel filterbank to power spectrogram
    mel_spec = apply_mel_filterbank(power_spec, filterbank)

    result = {'melspectrogram': mel_spec}

    if return_metadata:
        # Calculate frame times manually: (frame_index * hop_length + offset) / sr
        # Account for center padding offset
        n_frames = mel_spec.shape[1]
        frame_offset = frame_size // 2  # Center padding offset
        times = (np.arange(n_frames) * hop_length + frame_offset) / sr
        return {'melspectrogram': mel_spec, 'metadata': {'times': times}}

    return result
