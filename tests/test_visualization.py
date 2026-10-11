"""
Unit tests for afx_audio.utils.visualization.
"""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from afx_audio.utils.visualization import (
    plot_feature_distribution,
    plot_mfcc,
    plot_spectrogram,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def test_plot_mfcc_returns_axes():
    mfcc = np.random.RandomState(0).randn(13, 20)
    ax = plot_mfcc(mfcc, sr=22050, show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_mfcc_on_existing_axes():
    mfcc = np.random.RandomState(0).randn(13, 20)
    _, ax_in = plt.subplots()
    ax_out = plot_mfcc(mfcc, sr=22050, ax=ax_in, show=False)
    assert ax_out is ax_in


def test_plot_mfcc_empty_raises():
    with pytest.raises(ValueError):
        plot_mfcc(np.empty((13, 0)), sr=22050, show=False)


def test_plot_spectrogram_log_returns_axes():
    S = np.abs(np.random.RandomState(1).randn(65, 30)) ** 2
    ax = plot_spectrogram(S, sr=22050, y_axis='log', show=False)
    assert isinstance(ax, plt.Axes)
    assert ax.get_yscale() == 'symlog'


def test_plot_spectrogram_linear_returns_axes():
    S = np.abs(np.random.RandomState(1).randn(65, 30)) ** 2
    ax = plot_spectrogram(S, sr=22050, y_axis='linear', show=False)
    assert isinstance(ax, plt.Axes)
    assert ax.get_yscale() == 'linear'


def test_plot_spectrogram_empty_raises():
    with pytest.raises(ValueError):
        plot_spectrogram(np.empty((65, 0)), sr=22050, show=False)


def test_plot_feature_distribution_returns_axes():
    feature = np.random.RandomState(2).randn(100)
    ax = plot_feature_distribution(feature, show=False)
    assert isinstance(ax, plt.Axes)


def test_plot_feature_distribution_empty_raises():
    with pytest.raises(ValueError):
        plot_feature_distribution(np.array([]), show=False)


@pytest.mark.parametrize('center', [True, False])
def test_plot_columns_sit_at_metadata_times(center):
    from afx_audio.extractors.cepstral_features import extract_melspectrogram, extract_mfcc

    sr, frame_size, hop = 16000, 512, 128
    signal = np.random.RandomState(3).randn(sr // 4)
    kw = dict(frame_size=frame_size, hop_length=hop, center=center, return_metadata=True)
    mfcc = extract_mfcc(signal, sr, **kw)
    ax = plot_mfcc(mfcc['mfcc'], sr, hop_length=hop, frame_size=frame_size, center=center, show=False)
    left, right = ax.images[0].get_extent()[:2]
    n = mfcc['mfcc'].shape[1]
    np.testing.assert_allclose(left + (np.arange(n) + 0.5) * (right - left) / n, mfcc['metadata']['times'])

    mel = extract_melspectrogram(signal, sr, **kw)
    ax = plot_spectrogram(mel['melspectrogram'], sr, hop_length=hop, frame_size=frame_size, center=center,
                          show=False)
    edges = ax.collections[0].get_coordinates()[0, :, 0]
    np.testing.assert_allclose(0.5 * (edges[:-1] + edges[1:]), mel['metadata']['times'])


def test_plot_spectrogram_unknown_y_axis_raises():
    with pytest.raises(ValueError, match='y_axis'):
        plot_spectrogram(np.ones((65, 4)), sr=22050, y_axis='Mel', show=False)


@pytest.mark.parametrize('fmin, fmax', [(0.0, None), (300.0, 8000.0)])
def test_plot_spectrogram_mel_rows_at_filter_centers(fmin, fmax):
    from afx_audio.methods.mel import hz_to_mel, mel_frequencies

    sr, n_mels = 22050, 40
    ax = plot_spectrogram(np.ones((n_mels, 10)), sr, y_axis='mel', fmin=fmin, fmax=fmax, show=False)
    edges = ax.collections[0].get_coordinates()[:, 0, 1]
    centers = mel_frequencies(n_mels, fmin, fmax, sr)[1:-1]
    np.testing.assert_allclose(0.5 * (hz_to_mel(edges[:-1]) + hz_to_mel(edges[1:])), hz_to_mel(centers))
    assert ax.get_yscale() == 'function'
    lo, hi = ax.get_ylim()
    ticks = ax.get_yticks()
    assert len(ticks) >= 3 and np.all((ticks >= lo) & (ticks <= hi))


@pytest.mark.parametrize('f0', [150.0, 1000.0, 5000.0])
def test_plot_spectrogram_mel_tone_lands_in_its_row(f0):
    from afx_audio.extractors.cepstral_features import extract_melspectrogram

    sr = 22050
    t = np.arange(sr // 2) / sr
    mel = extract_melspectrogram(np.sin(2 * np.pi * f0 * t), sr, n_mels=64)['melspectrogram']
    ax = plot_spectrogram(mel, sr, y_axis='mel', show=False)
    edges = ax.collections[0].get_coordinates()[:, 0, 1]
    row = int(np.argmax(mel.mean(axis=1)))
    assert edges[row] <= f0 <= edges[row + 1]
