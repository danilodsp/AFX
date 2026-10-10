"""
afx_audio.utils.framewise.framewise_extractor: stacking, framing and errors.
"""
import numpy as np
import pytest

from afx_audio.utils.framewise import framewise_extractor

SR = 8000


@framewise_extractor
def frame_mean(signal, sr=None):
    return {'frame_mean': np.mean(signal)}


@framewise_extractor
def first_and_last(signal, sr=None):
    return {'ends': np.array([signal[0], signal[-1]])}


@framewise_extractor
def column(signal, sr=None):
    return {'column': np.array([[signal[0]], [signal[-1]]])}


def test_scalar_results_stack_to_frames():
    x = np.repeat([1.0, 2.0, 3.0], 4)
    out = frame_mean(x, SR, frame_size=4, hop_length=4, center=False)
    np.testing.assert_array_equal(out['frame_mean'], [1.0, 2.0, 3.0])


def test_vector_results_stack_on_the_last_axis():
    x = np.arange(12, dtype=np.float64)
    out = first_and_last(x, SR, frame_size=4, hop_length=4, center=False)
    np.testing.assert_array_equal(out['ends'], [[0, 4, 8], [3, 7, 11]])


def test_trailing_singleton_axis_is_dropped():
    x = np.arange(12, dtype=np.float64)
    out = column(x, SR, frame_size=4, hop_length=4, center=False)
    np.testing.assert_array_equal(out['column'], [[0, 4, 8], [3, 7, 11]])


def test_without_framing_the_whole_signal_is_one_frame():
    out = frame_mean(np.arange(10, dtype=np.float64), SR)
    np.testing.assert_array_equal(out['frame_mean'], [4.5])


def test_metadata_times_follow_the_framing():
    x = np.zeros(16)
    plain = frame_mean(x, SR, frame_size=4, hop_length=4, center=False, return_metadata=True)
    np.testing.assert_allclose(plain['metadata']['times'], (np.arange(4) * 4 + 2) / SR)
    centered = frame_mean(x, SR, frame_size=4, hop_length=4, return_metadata=True)
    np.testing.assert_allclose(centered['metadata']['times'], np.arange(5) * 4 / SR)


def test_metadata_needs_sr():
    with pytest.raises(ValueError, match='sr is required'):
        frame_mean(np.zeros(16), None, frame_size=4, hop_length=4, return_metadata=True)


def test_more_than_one_key_is_rejected():
    @framewise_extractor
    def two(signal, sr=None):
        return {'a': 1.0, 'b': 2.0, 'metadata': {}}

    with pytest.raises(ValueError, match=r"two returned 2 feature keys \['a', 'b'\]"):
        two(np.zeros(8), SR, frame_size=4, hop_length=4)


def test_parameters_reach_the_wrapped_function():
    @framewise_extractor
    def scaled_peak(signal, sr=None, *, scale=1.0):
        return {'peak': scale * np.max(np.abs(signal))}

    out = scaled_peak(np.array([0.0, -1.0, 0.5, 0.25]), SR, frame_size=2, hop_length=2, scale=3.0, center=False)
    np.testing.assert_array_equal(out['peak'], [3.0, 1.5])
