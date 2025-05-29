"""
Pitch detection utilities.
"""
import numpy as np


def yin(y: np.ndarray, frame_length: int, hop_length: int, fmin: float = 50.0,
        fmax: float = 2000.0, sr: int = 22050, threshold: float = 0.1):
    """
    YIN pitch detection algorithm implementation.

    Based on the algorithm described in "YIN, a fundamental frequency estimator for speech and music"
    by Alain de Cheveigné and Hideki Kawahara.

    Args:
        y: Input signal (1D np.ndarray)
        frame_length: Length of each analysis frame in samples
        hop_length: Number of samples between frames
        fmin: Minimum frequency to detect (Hz)
        fmax: Maximum frequency to detect (Hz)
        sr: Sampling rate (Hz)
        threshold: Threshold for pitch detection (default: 0.1)

    Returns:
        np.ndarray: Array of pitch values in Hz, shape (n_frames,)
    """
    # Frame the signal
    if len(y) < frame_length:
        return np.array([0.0])

    # Calculate minimum and maximum tau (lag) to search
    tau_min = int(sr / fmax) if fmax > 0 else 1
    tau_max = min(int(sr / fmin) if fmin > 0 else frame_length // 2, frame_length // 2)

    # Ensure we have valid range
    if tau_min >= tau_max:
        return np.zeros(1 + (len(y) - frame_length) // hop_length)

    # Create frames
    n_frames = 1 + (len(y) - frame_length) // hop_length
    pitches = np.zeros(n_frames)

    for frame_idx in range(n_frames):
        start = frame_idx * hop_length
        frame = y[start:start + frame_length]

        # Step 1: Calculate the difference function
        # Using the formula d_t(tau) = Σ(x_t(j) - x_t(j + tau))²
        difference = np.zeros(tau_max + 1)

        # Start with autocorrelation-like method for better efficiency
        for tau in range(1, tau_max + 1):
            difference[tau] = np.sum((frame[:frame_length-tau] - frame[tau:frame_length])**2)

        # Step 2: Calculate the cumulative mean normalized difference function
        cmnd = np.zeros_like(difference)
        cmnd[0] = 1.0  # By definition in the YIN paper

        # Calculate the cumulative sum for normalization
        cum_sum = np.cumsum(difference[1:])
        for tau in range(1, tau_max + 1):
            # Avoid division by zero
            if cum_sum[tau-1] > 0:
                cmnd[tau] = difference[tau] * tau / cum_sum[tau-1]
            else:
                cmnd[tau] = 1.0  # Set to 1 if normalization fails

        # Step 3: Find the first dip below threshold within valid tau range
        tau_pitch = tau_min
        found_pitch = False

        # Find the first local minimum below the threshold
        for tau in range(tau_min, tau_max):
            if cmnd[tau] < threshold:
                # Find the minimum in this valley
                local_min = tau
                while tau + 1 < tau_max and cmnd[tau+1] < cmnd[tau]:
                    tau += 1
                    if cmnd[tau] < cmnd[local_min]:
                        local_min = tau
                
                tau_pitch = local_min
                found_pitch = True
                break

        # If no dip is found below threshold, pick the minimum
        if not found_pitch:
            min_idx = np.argmin(cmnd[tau_min:tau_max+1])
            tau_pitch = min_idx + tau_min

        # Step 4: Refine the pitch estimation with parabolic interpolation
        if tau_pitch > 0 and tau_pitch < tau_max:
            # Ensure we have valid indices
            prev_tau = max(1, tau_pitch - 1)
            next_tau = min(tau_max - 1, tau_pitch + 1)
            
            # Parabolic interpolation
            if prev_tau > 0 and next_tau < tau_max:
                a = cmnd[prev_tau-1]
                b = cmnd[prev_tau]
                c = cmnd[next_tau]
                
                # Only use interpolation if we have a clear minimum
                if 2*b < a + c:  # Check if it's a minimum
                    offset = 0.5 * (a - c) / (a - 2*b + c)
                    tau_pitch = prev_tau + offset

        # Convert tau to frequency (Hz)
        if found_pitch or (tau_pitch < len(cmnd) and cmnd[int(tau_pitch)] < 0.5):  # 0.5 is a bit more generous than threshold
            pitches[frame_idx] = sr / tau_pitch if tau_pitch > 0 else 0.0
        else:
            pitches[frame_idx] = 0.0

    # Post-processing: Ensure frequencies are within valid range
    pitches[pitches < fmin] = 0.0
    pitches[pitches > fmax] = 0.0
            
    return pitches