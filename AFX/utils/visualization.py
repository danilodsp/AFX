"""
Visualization utilities for audio features.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_mfcc(mfcc: np.ndarray, sr: int, hop_length: int = 512, title: str = 'MFCC') -> None:
    """
    Plot a Mel-frequency cepstral coefficients (MFCC) matrix as an image.
    Args:
        mfcc: MFCC matrix (n_mfcc, n_frames)
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        title: Plot title
    """
    plt.figure(figsize=(10, 4))
    time_axis = np.arange(mfcc.shape[1]) * hop_length / sr
    extent = [time_axis[0], time_axis[-1], 0, mfcc.shape[0]]
    im = plt.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis', extent=extent)
    plt.colorbar(im, format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.ylabel('MFCC Coefficient')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(S: np.ndarray, sr: int, hop_length: int = 512, y_axis: str = 'log', title: str = 'Spectrogram') -> None:
    """
    Plot a spectrogram (in power or dB scale) as an image.
    Args:
        S: Spectrogram (frequency x time)
        sr: Sample rate
        hop_length: Hop length used in feature extraction
        y_axis: 'log' for log-frequency axis, 'linear' for linear
        title: Plot title
    """
    eps = 1e-10
    S_db = 10 * np.log10(S + eps)
    plt.figure(figsize=(10, 4))
    time_axis = np.arange(S.shape[1]) * hop_length / sr
    if y_axis == 'log':
        freq_axis = np.logspace(np.log10(1), np.log10(sr // 2), S.shape[0])
        extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        aspect = 'auto'
        plt.imshow(S_db, aspect=aspect, origin='lower', cmap='magma', extent=extent)
        plt.yscale('log')
        plt.ylabel('Frequency (Hz, log)')
    else:
        freq_axis = np.linspace(0, sr // 2, S.shape[0])
        extent = [time_axis[0], time_axis[-1], freq_axis[0], freq_axis[-1]]
        plt.imshow(S_db, aspect='auto', origin='lower', cmap='magma', extent=extent)
        plt.ylabel('Frequency (Hz)')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel('Time (s)')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_distribution(feature: np.ndarray, title: str = 'Feature Distribution') -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(feature.flatten(), bins=50, color='steelblue', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.show()
