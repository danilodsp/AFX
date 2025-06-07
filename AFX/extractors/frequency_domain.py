import numpy as np
from typing import Dict
from AFX.utils.framewise import framewise_extractor

__all__ = [
    'extract_spectral_centroid',
    'extract_spectral_bandwidth',
    'extract_spectral_rolloff',
    'extract_spectral_contrast',
    'extract_spectral_entropy',
    'extract_spectral_flatness',
    'extract_spectral_flux',
    'extract_spectral_skewness',
    'extract_spectral_slope',
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

    Implementation:
        Computes spectral centroid as the weighted mean of frequencies 
        using magnitude spectrum as weights.
    """
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    fft = np.fft.rfft(windowed_signal, n=frame_size)
    magnitude = np.abs(fft)

    freqs = np.fft.rfftfreq(frame_size, d=1.0/sr)

    # Compute the spectral centroid as weighted mean
    magnitude_sum = np.sum(magnitude)
    if magnitude_sum > 0:
        centroid = np.sum(freqs * magnitude) / magnitude_sum
    else:
        centroid = 0.0

    return {'spectral_centroid': np.array([centroid])}


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

    Spectral bandwidth is the weighted standard deviation of frequencies around the spectral centroid.

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

    Implementation:
        Computes spectral bandwidth using the formula:
        sqrt(sum(((freqs - centroid)^2) * magnitudes) / sum(magnitudes))
    """
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    fft = np.fft.rfft(windowed_signal, n=frame_size)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(frame_size, d=1.0/sr)

    # Calculate the spectral centroid first
    magnitude_sum = np.sum(magnitude)
    if magnitude_sum > 0:
        centroid = np.sum(freqs * magnitude) / magnitude_sum

        # Calculate the spectral bandwidth as weighted standard deviation
        # sqrt(sum(((freqs - centroid)^2) * magnitudes) / sum(magnitudes))
        bandwidth = np.sqrt(np.sum(((freqs - centroid)**2) * magnitude) / magnitude_sum)
    else:
        bandwidth = 0.0

    return {'spectral_bandwidth': np.array([bandwidth])}


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

    Spectral rolloff is the frequency below which a specified percentage 
    of the total spectral energy is contained.

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

    Implementation:
        Computes spectral rolloff using the formula:
        1. Compute power spectrum (magnitude squared)
        2. Calculate cumulative sum of power
        3. Normalize to get cumulative energy ratio
        4. Find frequency bin where cumulative energy >= roll_percent
    """
    window = np.hanning(len(signal))
    windowed_signal = signal * window

    fft = np.fft.rfft(windowed_signal, n=frame_size)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(frame_size, d=1.0/sr)

    # Compute power spectrum
    power = magnitude ** 2

    # Calculate total energy
    total_energy = np.sum(power)

    if total_energy > 0:
        # Calculate cumulative energy
        cumulative_energy = np.cumsum(power)

        # Normalize to get cumulative energy ratio
        cumulative_ratio = cumulative_energy / total_energy

        # Find the first frequency bin where cumulative energy >= roll_percent
        rolloff_idx = np.where(cumulative_ratio >= roll_percent)[0]

        if len(rolloff_idx) > 0:
            # Use linear interpolation for more precise rolloff frequency
            idx = rolloff_idx[0]
            if idx > 0:
                # Interpolate between idx-1 and idx
                energy_before = cumulative_energy[idx-1]
                energy_after = cumulative_energy[idx]
                freq_before = freqs[idx-1] 
                freq_after = freqs[idx]

                target_energy = roll_percent * total_energy

                # Linear interpolation
                if energy_after > energy_before:
                    alpha = (target_energy - energy_before) / (energy_after - energy_before)
                    rolloff_frequency = freq_before + alpha * (freq_after - freq_before)
                else:
                    rolloff_frequency = freq_after
            else:
                rolloff_frequency = freqs[idx]
        else:
            # If no bin exceeds the threshold, use the highest frequency
            rolloff_frequency = freqs[-1]
    else:
        # If no energy, rolloff is 0
        rolloff_frequency = 0.0

    return {'spectral_rolloff': np.array([rolloff_frequency])}

def extract_spectral_contrast(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    n_bands: int = 6,
    fmin: float = 200.0,
    quantile: float = 0.02,
    return_metadata: bool = False,
    **kwargs
) -> Dict[str, np.ndarray]:
    """
    Compute the spectral contrast of an audio signal.

    Spectral contrast measures the difference in amplitude between peaks and valleys
    in the spectrum. Each frame of a spectrogram is divided into sub-bands.
    For each sub-band, the energy contrast is estimated by comparing
    the mean energy in the top quantile (peak energy) to that of the
    bottom quantile (valley energy).

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        n_bands: Number of frequency bands
        fmin: Minimum frequency for the first band
        quantile: Quantile for determining peaks and valleys
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_contrast' key and np.ndarray of values
    Metadata:
        shape: (n_bands + 1, n_frames)
        units: dB
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Implementation:
        1. Compute magnitude spectrogram using STFT
        2. Divide frequency into logarithmically spaced bands
        3. For each band and frame:
           - Find the top and bottom quantile of magnitudes
           - Calculate contrast as the difference between top and bottom in dB
    """
    window = np.hanning(frame_size)

    # Pad the signal for center framing
    pad_width = int(frame_size // 2)
    signal_padded = np.pad(signal, (pad_width, pad_width), mode='constant')

    # Number of frames
    n_frames = 1 + (len(signal_padded) - frame_size) // hop_length

    # Compute the STFT
    S = np.zeros((frame_size // 2 + 1, n_frames), dtype=complex)
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_size
        frame = signal_padded[start:end] * window
        S[:, i] = np.fft.rfft(frame, n=frame_size)

    # Get magnitude spectrum
    S_mag = np.abs(S)

    # Compute frequency bins
    freq = np.fft.rfftfreq(frame_size, d=1.0/sr)

    # Calculate logarithmically spaced bands
    octa = np.zeros(n_bands + 2)
    octa[1:] = fmin * (2.0 ** np.arange(0, n_bands + 1))

    # Check if any band exceeds Nyquist
    nyquist = sr / 2.0
    if np.any(octa[:-1] >= nyquist):
        raise ValueError("Frequency band exceeds Nyquist. Reduce fmin or n_bands.")

    # Initialize arrays for peaks and valleys
    valley = np.zeros((n_bands + 1, n_frames))
    peak = np.zeros((n_bands + 1, n_frames))

    # For each band, compute peak and valley
    for k in range(n_bands + 1):
        f_low, f_high = octa[k], octa[k + 1]

        # Find indices for this band
        current_band = np.logical_and(freq >= f_low, freq <= f_high)
        idx = np.flatnonzero(current_band)

        if len(idx) == 0:
            continue

        # Handle band boundaries
        if k > 0 and idx[0] > 0:
            current_band[idx[0] - 1] = True
            idx = np.flatnonzero(current_band)

        if k == n_bands:
            current_band[idx[-1] + 1:] = True
            idx = np.flatnonzero(current_band)

        # Get sub-band
        sub_band = S_mag[current_band]

        if k < n_bands and len(sub_band) > 0:
            # Exclude highest bin from all but last band
            sub_band = sub_band[:-1]

        if len(sub_band) == 0:
            continue

        # Always take at least one bin
        n_quantile = int(np.maximum(np.rint(quantile * len(sub_band)), 1))

        # For each frame
        for j in range(n_frames):
            frame = sub_band[:, j]

            if len(frame) == 0:
                continue

            # Sort magnitudes
            sorted_frame = np.sort(frame)

            # Compute valleys and peaks
            valley[k, j] = np.mean(sorted_frame[:n_quantile])
            peak[k, j] = np.mean(sorted_frame[-n_quantile:])

    # Small constant to avoid log(0)
    eps = 1e-10

    # Convert to dB scale
    peak_db = 20 * np.log10(peak + eps)
    valley_db = 20 * np.log10(valley + eps)

    # Compute contrast as difference in dB
    contrast = peak_db - valley_db

    result = {'spectral_contrast': contrast}
    if return_metadata:
        times = np.arange(n_frames) * hop_length / sr
        result['metadata'] = {'times': times}
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

    Spectral entropy measures the flatness or peakiness of the power spectrum. It is higher for
    noise-like signals with uniform energy distribution and lower for tonal signals where
    energy is concentrated in specific frequency bins.

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

    Mathematical formulation:
        1. Compute the power spectrum P(f) from STFT
        2. Normalize P(f) to a probability distribution: p(f) = P(f) / sum(P(f))
        3. Calculate entropy: H = -sum(p(f) * log2(p(f)))

    Implementation:
        1. Apply windowing to the signal using a Hann window
        2. Compute the STFT and magnitude spectrum
        3. Normalize the power spectrum to a probability distribution
        4. Calculate the spectral entropy for each frame
    """
    # Apply windowing (hann window)
    window = np.hanning(frame_size)

    # Compute STFT - use rfft for real-valued signals
    n_frames = 1 + (len(signal) - frame_size) // hop_length

    # Initialize output array
    entropy = np.zeros(n_frames)

    # Small constant to avoid log(0)
    eps = 1e-10

    # Process each frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_size

        # Apply windowing to the frame
        if end <= len(signal):
            frame = signal[start:end] * window
        else:
            # Pad the last frame if needed
            padding = end - len(signal)
            frame = np.pad(signal[start:], (0, padding)) * window

        # Compute FFT
        fft = np.fft.rfft(frame, n=frame_size)

        # Compute the magnitude spectrum
        magnitude = np.abs(fft)

        # Normalize the power spectrum to get a probability distribution
        power_sum = np.sum(magnitude)
        if power_sum > eps:
            # Normalize to probability distribution
            ps = magnitude / power_sum
            # Calculate entropy
            entropy[i] = -np.sum(ps * np.log2(ps + eps))
        else:
            entropy[i] = 0.0

    result = {'spectral_entropy': entropy}

    if return_metadata:
        # Calculate frame times based on the hop length and sample rate
        times = np.arange(n_frames) * hop_length / sr
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

    Spectral flatness measures the noisiness or tonality of a sound and is defined as the ratio
    of the geometric mean to the arithmetic mean of the power spectrum.

    Values close to 1.0 indicate a noise-like signal, while values close to 0.0 indicate tonal content.

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
        units: float (range from 0 to 1, where 1 is perfectly flat spectrum)
        times: np.ndarray, shape (n_frames,) if return_metadata is True

    Implementation:
        1. Apply windowing to the signal using a Hann window
        2. Compute the STFT and magnitude spectrum
        3. Calculate the ratio of geometric mean to arithmetic mean of the spectrum for each frame
    """
    window = np.hanning(frame_size)

    # Compute the FFT - use rfft for real-valued signals
    n_frames = 1 + (len(signal) - frame_size) // hop_length

    # Initialize output array
    flatness = np.zeros(n_frames)

    # Small constant to avoid log(0) and division by zero
    eps = 1e-10

    # Process each frame
    for i in range(n_frames):
        start = i * hop_length
        end = start + frame_size

        # Apply windowing to the frame
        if end <= len(signal):
            frame = signal[start:end] * window
        else:
            # Pad the last frame if needed
            padding = end - len(signal)
            frame = np.pad(signal[start:], (0, padding)) * window

        # Compute FFT
        fft = np.fft.rfft(frame, n=frame_size)

        # Compute the power spectrum (squared magnitude)
        power_spectrum = np.abs(fft) ** 2

        # Ensure spectrum is positive for log calculation
        power_spectrum = np.maximum(power_spectrum, eps)

        # Compute geometric mean: exp(mean(log(spectrum)))
        geometric_mean = np.exp(np.mean(np.log(power_spectrum)))

        # Compute arithmetic mean: mean(spectrum)
        arithmetic_mean = np.mean(power_spectrum)

        # Compute flatness as the ratio of geometric to arithmetic mean
        if arithmetic_mean > eps:
            flatness[i] = geometric_mean / arithmetic_mean
        else:
            flatness[i] = 0.0

    result = {'spectral_flatness': flatness}

    if return_metadata:
        # Calculate frame times based on the hop length and sample rate
        times = np.arange(n_frames) * hop_length / sr
        return {'spectral_flatness': flatness, 'metadata': {'times': times}}

    return result

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

    Spectral flux measures the frame-to-frame change in the magnitude spectrum, 
    computed as the Euclidean distance between consecutive magnitude spectra.

    Args:
        signal: Audio signal (1D np.ndarray)
        sr: Sample rate
        frame_size: Frame size for STFT
        hop_length: Hop length between frames
        return_metadata: If True, include frame times in metadata
    Returns:
        Dict with 'spectral_flux' key and np.ndarray of values
    Metadata:
        shape: (n_frames-1,)
        units: float
        times: np.ndarray, shape (n_frames-1,) if return_metadata is True

    Implementation:
        1. Compute magnitude spectrogram using NumPy-based STFT
        2. Calculate the Euclidean distance between consecutive magnitude spectra:
           flux[t] = sqrt(sum((|X_t| - |X_{t-1}|)^2)) for t > 0
    """
    # Create a Hann window
    window = np.hanning(frame_size)

    # Compute direct STFT using NumPy

    # Step 1: Calculate number of frames and prepare output array
    n_samples = len(signal)

    # Centering the frames
    # We need to pad the signal at both ends
    pad_width = int(frame_size // 2)
    signal_padded = np.pad(signal, pad_width, mode='reflect')
    n_padded = len(signal_padded)

    # Calculate number of frames
    n_frames = 1 + (n_padded - frame_size) // hop_length

    # Prepare STFT output array - only need the positive frequencies for real signal
    n_freqs = frame_size // 2 + 1
    stft_matrix = np.empty((n_freqs, n_frames), dtype=np.complex64)

    # Step 2: Compute STFT frame by frame
    for i in range(n_frames):
        # Extract the frame
        frame_start = i * hop_length
        frame_end = frame_start + frame_size
        frame = signal_padded[frame_start:frame_end]

        # Apply window
        windowed_frame = frame * window

        # Compute FFT
        stft_matrix[:, i] = np.fft.rfft(windowed_frame, n=frame_size)

    # Step 3: Compute magnitude spectrogram
    magnitude = np.abs(stft_matrix)

    # Step 4: Compute spectral flux
    # Compute differences between adjacent spectral frames
    diffs = np.diff(magnitude, axis=1)

    # Compute Euclidean distance (sqrt of sum of squares)
    flux = np.sqrt(np.sum(diffs ** 2, axis=0))

    # Return results
    result = {'spectral_flux': flux}

    if return_metadata:
        # Calculate frame times based on hop length and sample rate
        # Note: flux has one fewer frame than the STFT because of the diff operation
        # Match frames_to_time which accounts for center=True by adding frame_size/2 to the time
        # This is because the center of the first frame is at frame_size/2 samples
        # and then each hop adds hop_length samples
        times = (np.arange(len(flux)) * hop_length + frame_size / 2) / sr
        result['metadata'] = {'times': times}

    return result


@framewise_extractor
def extract_spectral_skewness(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute the spectral skewness of an audio signal."""
    window = np.hanning(frame_size)
    windowed_signal = signal[:frame_size] * window

    fft = np.fft.rfft(windowed_signal, n=frame_size)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    mag_sum = np.sum(magnitude)
    if mag_sum > 0:
        centroid = np.sum(freqs * magnitude) / mag_sum
        bandwidth = np.sqrt(
            np.sum(((freqs - centroid) ** 2) * magnitude) / mag_sum
        )
        if bandwidth > 0:
            skewness = (
                np.sum(((freqs - centroid) / bandwidth) ** 3 * magnitude)
                / mag_sum
            )
        else:
            skewness = 0.0
    else:
        skewness = 0.0

    return {'spectral_skewness': np.array([skewness])}


@framewise_extractor
def extract_spectral_slope(
    signal: np.ndarray,
    sr: int,
    frame_size: int = 2048,
    hop_length: int = 512,
    return_metadata: bool = False,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """Compute the spectral slope of an audio signal."""
    window = np.hanning(frame_size)
    windowed_signal = signal[:frame_size] * window

    fft = np.fft.rfft(windowed_signal, n=frame_size)
    magnitude = np.abs(fft)
    freqs = np.fft.rfftfreq(frame_size, d=1.0 / sr)

    eps = 1e-10
    mag_db = 20 * np.log10(magnitude + eps)

    x = freqs
    y = mag_db
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    denom = np.sum((x - x_mean) ** 2)
    if denom > 0:
        slope = np.sum((x - x_mean) * (y - y_mean)) / denom
    else:
        slope = 0.0

    return {'spectral_slope': np.array([slope])}
