"""
Visualization utilities for audio features.
"""
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import librosa.display

def plot_mfcc(mfcc: np.ndarray, sr: int, hop_length: int = 512, title: str = 'MFCC') -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr, hop_length=hop_length, cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_spectrogram(S: np.ndarray, sr: int, hop_length: int = 512, y_axis: str = 'log', title: str = 'Spectrogram') -> None:
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max), x_axis='time', y_axis=y_axis, sr=sr, hop_length=hop_length, cmap='magma')
    plt.colorbar(format='%+2.0f dB')
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
