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
