"""
Visualization utilities for audio features.
"""

from typing import cast

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.ticker import FixedLocator, NullLocator, ScalarFormatter
except ImportError as e:
    raise ImportError('afx_audio.utils.visualization requires matplotlib: pip install "afx-audio[plot]"') from e
import numpy as np

from afx_audio.methods.mel import hz_to_mel, mel_to_hz
from afx_audio.utils.frame_utils import frame_times


def plot_mfcc(
    mfcc: np.ndarray,
    sr: int,
    hop_length: int = 512,
    title: str = 'MFCC',
    ax: plt.Axes | None = None,
    show: bool = True,
    *,
    frame_size: int = 2048,
    center: bool = True,
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
        frame_size: Frame size used in feature extraction
        center: Whether the frames were centered
    Returns:
        The matplotlib Axes containing the plot, with each column drawn at
        the frame time the extractors report in ``metadata['times']``.
    """
    mfcc = np.asarray(mfcc)
    if mfcc.ndim != 2 or mfcc.shape[1] == 0 or mfcc.shape[0] == 0:
        raise ValueError(
            f'mfcc must be a non-empty 2D array (n_mfcc, n_frames), got shape {mfcc.shape}'
        )
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    time_axis = frame_times(mfcc.shape[1], sr, hop_length, frame_size, center)
    # Each column spans one hop around its frame time
    half_hop = 0.5 * hop_length / sr
    extent = (float(time_axis[0]) - half_hop, float(time_axis[-1]) + half_hop, 0.0, float(mfcc.shape[0]))
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


def _mel_ticks(lo: float, hi: float) -> list[float]:
    """Octave ticks from 125 Hz for a mel axis, thinned to at least 1/10 of the axis apart."""
    # The default linear-Hz ticks leave the lower half of a mel axis unlabeled
    candidates = [125.0 * 2**k for k in range(10)]
    min_gap = (hz_to_mel(hi) - hz_to_mel(lo)) / 10
    ticks: list[float] = []
    for f in candidates:
        if lo <= f <= hi and (not ticks or hz_to_mel(f) - hz_to_mel(ticks[-1]) >= min_gap):
            ticks.append(f)
    return ticks


def plot_spectrogram(
    S: np.ndarray,
    sr: int,
    hop_length: int = 512,
    y_axis: str = 'log',
    title: str = 'Spectrogram',
    ax: plt.Axes | None = None,
    show: bool = True,
    *,
    frame_size: int = 2048,
    center: bool = True,
    fmin: float = 0.0,
    fmax: float | None = None,
) -> plt.Axes:
    """
    Plot a spectrogram (in power scale, displayed in dB) as an image.

    Args:
        S: Spectrogram (frequency x time), linear power values
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        y_axis: 'log' or 'linear' for an STFT spectrogram, 'mel' for a mel
            spectrogram such as extract_melspectrogram's output
        title: Plot title
        ax: Existing matplotlib Axes to draw on. If None, a new figure and
            axes are created.
        show: Whether to call plt.show(). Only honored when a new figure was
            created (ax was None).
        frame_size: Frame size (n_fft) used in feature extraction
        center: Whether the frames were centered
        fmin: Lowest mel band edge in Hz, as passed to the extractor
            (y_axis='mel' only)
        fmax: Highest mel band edge in Hz, default sr / 2 (y_axis='mel' only)
    Returns:
        The matplotlib Axes containing the plot, with each column drawn at
        the frame time the extractors report in ``metadata['times']``.

    Notes:
        Frequency rows of an STFT spectrogram are linearly spaced bin centers
        (np.linspace(0, sr/2, n_freq)). For the log-frequency view the data is
        drawn with pcolormesh against those true linear bin frequencies and
        the axis is then log-scaled (symlog, so the DC bin at 0 Hz remains
        visible). Using imshow with a log-spaced extent would misplace bins.

        Row m of a mel spectrogram is the filter centered at
        mel_frequencies(n_mels, fmin, fmax, sr)[m + 1] (Slaney scale). With
        y_axis='mel' each row spans half a mel step either side of that
        center and the axis is mel-scaled, so all rows have the same height.
    """
    S = np.asarray(S)
    if S.ndim != 2 or S.shape[1] == 0 or S.shape[0] == 0:
        raise ValueError(
            f'S must be a non-empty 2D array (n_freq, n_frames), got shape {S.shape}'
        )
    if y_axis not in ('log', 'linear', 'mel'):
        raise ValueError(f"y_axis must be 'log', 'linear' or 'mel', got {y_axis!r}")
    eps = 1e-10
    S_db = 10 * np.log10(S + eps)
    created_fig = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))
    time_axis = frame_times(S.shape[1], sr, hop_length, frame_size, center)
    if y_axis == 'mel':
        mel_lo, mel_hi = hz_to_mel(np.array([fmin, sr / 2.0 if fmax is None else fmax]))
        step = (mel_hi - mel_lo) / (S.shape[0] + 1)
        # Filter centers are equally spaced in mel; row edges sit halfway between them
        freq_edges = mel_to_hz(mel_lo + step * (np.arange(S.shape[0] + 1) + 0.5))
        half_hop = 0.5 * hop_length / sr
        time_edges = np.append(time_axis - half_hop, time_axis[-1] + half_hop)
        mesh = ax.pcolormesh(time_edges, freq_edges, S_db, cmap='magma', shading='flat')
        ax.set_yscale('function', functions=(hz_to_mel, mel_to_hz))
        ax.set_ylim(freq_edges[0], freq_edges[-1])
        ax.yaxis.set_major_locator(FixedLocator(_mel_ticks(freq_edges[0], freq_edges[-1])))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.yaxis.set_minor_locator(NullLocator())
        ax.set_ylabel('Frequency (Hz, mel)')
    else:
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
