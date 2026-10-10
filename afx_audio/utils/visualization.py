"""
Visualization utilities for audio features.
"""

from typing import cast

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
except ImportError as e:
    raise ImportError('afx_audio.utils.visualization requires matplotlib: pip install "afx-audio[plot]"') from e
import numpy as np


def plot_mfcc(
    mfcc: np.ndarray,
    sr: int,
    hop_length: int = 512,
    title: str = 'MFCC',
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a Mel-frequency cepstral coefficients (MFCC) matrix as an image.

    Args:
        mfcc: MFCC matrix (n_mfcc, n_frames)
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        title: Plot title
        ax: Existing matplotlib Axes to draw on. If None, a new figure and
            axes are created.
        show: Whether to call plt.show(). Only honored when a new figure was
            created (ax was None).
    Returns:
        The matplotlib Axes containing the plot.
    """
    mfcc = np.asarray(mfcc)
    if mfcc.ndim != 2 or mfcc.shape[1] == 0 or mfcc.shape[0] == 0:
        raise ValueError(
            f'mfcc must be a non-empty 2D array (n_mfcc, n_frames), got shape {mfcc.shape}'
        )
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    time_axis = np.arange(mfcc.shape[1]) * hop_length / sr
    extent = (float(time_axis[0]), float(time_axis[-1]), 0.0, float(mfcc.shape[0]))
    im = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    # MFCCs are dimensionless coefficients, not dB
    ax.figure.colorbar(im, ax=ax)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_title(title)
    cast(Figure, ax.figure).tight_layout()
    if show and created_fig:
        plt.show()
    return ax


def plot_spectrogram(
    S: np.ndarray,
    sr: int,
    hop_length: int = 512,
    y_axis: str = 'log',
    title: str = 'Spectrogram',
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a spectrogram (in power scale, displayed in dB) as an image.

    Args:
        S: Spectrogram (frequency x time), linear power values
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        y_axis: 'log' for log-frequency axis, 'linear' for linear
        title: Plot title
        ax: Existing matplotlib Axes to draw on. If None, a new figure and
            axes are created.
        show: Whether to call plt.show(). Only honored when a new figure was
            created (ax was None).
    Returns:
        The matplotlib Axes containing the plot.

    Notes:
        Frequency rows of an STFT spectrogram are linearly spaced bin centers
        (np.linspace(0, sr/2, n_freq)). For the log-frequency view the data is
        drawn with pcolormesh against those true linear bin frequencies and
        the axis is then log-scaled (symlog, so the DC bin at 0 Hz remains
        visible). Using imshow with a log-spaced extent would misplace bins.
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[1] == 0 or S.shape[0] == 0:
        raise ValueError(
            f'S must be a non-empty 2D array (n_freq, n_frames), got shape {S.shape}'
        )
    eps = 1e-10
    S_db = 10 * np.log10(S + eps)
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    time_axis = np.arange(S.shape[1]) * hop_length / sr
    # True linear bin center frequencies for the rows of S
    freq_axis = np.linspace(0, sr / 2, S.shape[0])
    mesh = ax.pcolormesh(time_axis, freq_axis, S_db, cmap='magma', shading='auto')
    if y_axis == 'log':
        # symlog keeps the 0 Hz (DC) bin on-axis; linear below linthresh
        ax.set_yscale('symlog', linthresh=max(sr / S.shape[0], 1.0))
        ax.set_ylabel('Frequency (Hz, log)')
    else:
        ax.set_ylabel('Frequency (Hz)')
    ax.figure.colorbar(mesh, ax=ax, format='%+2.0f dB')
    ax.set_xlabel('Time (s)')
    ax.set_title(title)
    cast(Figure, ax.figure).tight_layout()
    if show and created_fig:
        plt.show()
    return ax


def plot_feature_distribution(
    feature: np.ndarray,
    title: str = 'Feature Distribution',
    ax: plt.Axes | None = None,
    show: bool = True,
) -> plt.Axes:
    """
    Plot a histogram of feature values.

    Args:
        feature: Feature array (any shape; flattened for the histogram)
        title: Plot title
        ax: Existing matplotlib Axes to draw on. If None, a new figure and
            axes are created.
        show: Whether to call plt.show(). Only honored when a new figure was
            created (ax was None).
    Returns:
        The matplotlib Axes containing the plot.
    """
    feature = np.asarray(feature)
    if feature.size == 0:
        raise ValueError('feature must be a non-empty array')
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    ax.hist(feature.flatten(), bins=50, color='steelblue', alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')
    cast(Figure, ax.figure).tight_layout()
    if show and created_fig:
        plt.show()
    return ax
