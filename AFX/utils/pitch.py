"""
Pitch detection utilities.
"""
import numpy as np
from AFX.methods.autocorrelation import autocorrelation


def _difference_function(frame: np.ndarray, tau_max: int) -> np.ndarray:
    """
    YIN difference function d(tau) for tau = 0..tau_max, computed via
    cumulative energies and one FFT-based autocorrelation:

        d(tau) = sum_{j=0}^{W-tau-1} (x[j] - x[j+tau])^2
               = E1(tau) + E2(tau) - 2 * r(tau)

    where E1/E2 are prefix/suffix energies of the overlapping segments and
    r(tau) the linear autocorrelation. Equivalent to the naive O(W * tau_max)
    loop but O(W log W).
    """
    w = len(frame)
    frame = frame.astype(np.float64)
    sq = frame ** 2
    # cumulative energy: cum[k] = sum(sq[:k])
    cum = np.concatenate(([0.0], np.cumsum(sq)))
    taus = np.arange(tau_max + 1)
    # E1(tau) = sum_{j=0}^{w-tau-1} x[j]^2 ;  E2(tau) = sum_{j=tau}^{w-1} x[j]^2
    e1 = cum[w - taus]
    e2 = cum[w] - cum[taus]
    acf = autocorrelation(frame, tau_max + 1)
    d = e1 + e2 - 2.0 * acf
    # Guard tiny negative values from float error
    return np.maximum(d, 0.0)


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

        # Step 1: Calculate the difference function (FFT-based, O(W log W))
        difference = _difference_function(frame, tau_max)

        # Step 2: Calculate the cumulative mean normalized difference function
        cmnd = np.ones_like(difference)
        cmnd[0] = 1.0  # By definition in the YIN paper

        # cmnd[tau] = d[tau] * tau / sum(d[1..tau]); 1.0 where the sum is 0
        cum_sum = np.cumsum(difference[1:])
        valid = cum_sum > 0
        taus = np.arange(1, tau_max + 1)
        cmnd[1:][valid] = (
            difference[1:][valid] * taus[valid] / cum_sum[valid]
        )

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
        # Fit the parabola through the samples at (tau-1, tau, tau+1),
        # centered on the detected minimum itself.
        if 0 < tau_pitch < tau_max:
            a = cmnd[tau_pitch - 1]
            b = cmnd[tau_pitch]
            c = cmnd[tau_pitch + 1]

            # Only use interpolation if we have a clear minimum
            denom = a - 2 * b + c
            if denom > 0:  # Convex: it's a minimum
                offset = 0.5 * (a - c) / denom
                tau_pitch = tau_pitch + offset

        # Convert tau to frequency (Hz)
        if found_pitch or (tau_pitch < len(cmnd) and cmnd[int(tau_pitch)] < 0.5):  # 0.5 is a bit more generous than threshold
            pitches[frame_idx] = sr / tau_pitch if tau_pitch > 0 else 0.0
        else:
            pitches[frame_idx] = 0.0

    # Post-processing: Ensure frequencies are within valid range
    pitches[pitches < fmin] = 0.0
    pitches[pitches > fmax] = 0.0
            
    return pitches