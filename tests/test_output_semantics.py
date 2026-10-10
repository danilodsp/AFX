"""Output conventions shared by every built-in feature: dtype, framing and silent frames."""
import inspect

import numpy as np
import pytest

import afx_audio

SR = 16000
FEATURES = sorted(afx_audio.FEATURE_MAP)
NOISE = np.random.default_rng(0).standard_normal(SR // 2) * 0.1


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('name', FEATURES)
def test_output_is_float64(name, dtype):
    out = afx_audio.FEATURE_MAP[name](NOISE.astype(dtype), SR, return_metadata=True)
    assert out[name].dtype == np.float64
    assert np.asarray(out['metadata']['times']).dtype == np.float64


@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_torch_output_follows_input_dtype(dtype_name):
    torch = pytest.importorskip('torch')
    from afx_audio.torch import functional as F_afx
    from afx_audio.torch.module import _TORCH_FEATURES

    dtype = getattr(torch, dtype_name)
    x = torch.as_tensor(NOISE, dtype=dtype)
    for name, func in _TORCH_FEATURES.items():
        assert func(x, SR).dtype == dtype, name
    assert F_afx.delta(F_afx.mfcc(x, SR)).dtype == dtype


FRAMED = [name for name in FEATURES if 'frame_size' in afx_audio.feature_info(name)['parameters']]


def test_framed_features_share_frames_and_times():
    # Centered by default everywhere, so one framing gives one time axis
    signal = np.random.default_rng(1).standard_normal(SR // 2)
    ref = np.arange(1 + len(signal) // 128) * 128 / SR
    assert len(FRAMED) == 45
    for name in FRAMED:
        out = afx_audio.FEATURE_MAP[name](signal, SR, frame_size=512, hop_length=128, return_metadata=True)
        # flux[i] compares frames i and i + 1 and carries the time of frame i + 1
        expected = ref[1:] if name == 'spectral_flux' else ref
        assert out[name].shape[-1] == len(expected), name
        np.testing.assert_allclose(out['metadata']['times'], expected, rtol=0, atol=1e-12, err_msg=name)


def test_torch_framing_defaults_match_numpy():
    pytest.importorskip('torch')
    from afx_audio.torch.module import _TORCH_FEATURES

    for name, func in _TORCH_FEATURES.items():
        torch_params = inspect.signature(func).parameters
        numpy_params = afx_audio.feature_info(name)['parameters']
        for key in ('frame_size', 'hop_length', 'center'):
            assert torch_params[key].default == numpy_params[key], (name, key)


@pytest.mark.parametrize('preset', afx_audio.list_presets())
def test_preset_features_with_one_framing_share_frame_counts(preset):
    config = afx_audio.load_preset(preset)
    sr = config['sample_rate']
    frames = afx_audio.extract_all_features(np.random.default_rng(2).standard_normal(sr), sr,
                                            {**config, 'preserve_shape': True})
    counts: dict[tuple, set] = {}
    for name, params in config['features'].items():
        defaults = afx_audio.feature_info(name)['parameters']
        framing = (params.get('frame_size', defaults.get('frame_size')),
                   params.get('hop_length', defaults.get('hop_length')))
        if None in framing:
            continue
        counts.setdefault(framing, set()).add(frames[name].shape[-1] + (name == 'spectral_flux'))
    assert all(len(n) == 1 for n in counts.values()), counts
