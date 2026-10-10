"""
Unit tests for AFX.utils.visualization.
"""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pytest

from AFX.utils.visualization import (
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
