"""
Tests for AFX.torch: numerical parity with the NumPy path, gradient flow,
batching, and the AFXFeatureModule wrapper.
"""
import numpy as np
import pytest

torch = pytest.importorskip('torch')

import importlib

from AFX.extractors import cepstral_features, frequency_domain, time_domain

# AFX.methods re-exports the stft *function*; get the module explicitly
np_stft_mod = importlib.import_module('AFX.methods.stft')
from AFX.torch import AFXFeatureModule, functional as F_afx, supported_features


SR = 22050
RNG = np.random.default_rng(42)


@pytest.fixture
def signal():
    """1 second of a chirp-ish tone + noise (deterministic)."""
    t = np.arange(SR) / SR
    sig = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.3 * np.sin(2 * np.pi * 1237 * t * (1 + 0.3 * t))
        + 0.05 * RNG.standard_normal(SR)
    ).astype(np.float64)
    return sig


def _t(x):
    return torch.as_tensor(x, dtype=torch.float64)


# ---------------------------------------------------------------------------
# Parity with the NumPy path
# ---------------------------------------------------------------------------

def test_stft_parity(signal):
    ref = np_stft_mod.stft(signal, frame_size=1024, hop_length=256,
                           window='hann', center=True)
    out = F_afx.stft(_t(signal), frame_size=1024, hop_length=256,
                     center=True).numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


def test_stft_no_center_parity(signal):
    ref = np_stft_mod.stft(signal, frame_size=2048, hop_length=512,
                           window='hann', center=False)
    out = F_afx.stft(_t(signal), frame_size=2048, hop_length=512,
                     center=False).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-8, atol=1e-8)


def test_mfcc_parity(signal):
    ref = cepstral_features.extract_mfcc(
        signal, SR, n_mfcc=13, frame_size=1024, hop_length=256
    )['mfcc']
    out = F_afx.mfcc(_t(signal), SR, n_mfcc=13, frame_size=1024,
                     hop_length=256).numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6)


def test_melspectrogram_parity(signal):
    ref = cepstral_features.extract_melspectrogram(
        signal, SR, n_mels=64, frame_size=1024, hop_length=256
    )['melspectrogram']
    out = F_afx.melspectrogram(_t(signal), SR, n_mels=64, frame_size=1024,
                               hop_length=256).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('name,np_fn,torch_fn', [
    ('spectral_centroid', frequency_domain.extract_spectral_centroid,
     F_afx.spectral_centroid),
    ('spectral_bandwidth', frequency_domain.extract_spectral_bandwidth,
     F_afx.spectral_bandwidth),
    ('spectral_rolloff', frequency_domain.extract_spectral_rolloff,
     F_afx.spectral_rolloff),
    ('spectral_entropy', frequency_domain.extract_spectral_entropy,
     F_afx.spectral_entropy),
    ('spectral_flatness', frequency_domain.extract_spectral_flatness,
     F_afx.spectral_flatness),
    ('spectral_flux', frequency_domain.extract_spectral_flux,
     F_afx.spectral_flux),
])
def test_spectral_feature_parity(signal, name, np_fn, torch_fn):
    ref = np_fn(signal, SR, frame_size=1024, hop_length=256)[name]
    out = torch_fn(_t(signal), SR, frame_size=1024, hop_length=256).numpy()
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-8)


def test_zcr_parity(signal):
    ref = time_domain.extract_zero_crossing_rate(
        signal, SR, frame_size=1024, hop_length=256
    )['zcr']
    out = F_afx.zero_crossing_rate(_t(signal), SR, frame_size=1024,
                                   hop_length=256).numpy()
    np.testing.assert_allclose(out.ravel(), np.asarray(ref).ravel(),
                               rtol=1e-8, atol=1e-8)


def test_rms_parity(signal):
    ref = time_domain.extract_rms_energy(signal, SR)['rms_energy']
    out = F_afx.rms_energy(_t(signal), SR).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-10)


def test_dct_matrix_matches_scipy():
    from scipy.fftpack import dct
    x = RNG.standard_normal((40, 7))
    ref = dct(x, type=2, norm='ortho', axis=0)[:13]
    mat = F_afx.dct_matrix(13, 40, dtype=torch.float64)
    out = (mat @ torch.as_tensor(x)).numpy()
    np.testing.assert_allclose(out, ref, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Batching
# ---------------------------------------------------------------------------

def test_batched_matches_unbatched(signal):
    sig2 = signal[::-1].copy()
    batch = torch.stack([_t(signal), _t(sig2)])
    out_b = F_afx.mfcc(batch, SR, frame_size=1024, hop_length=256)
    out_0 = F_afx.mfcc(_t(signal), SR, frame_size=1024, hop_length=256)
    out_1 = F_afx.mfcc(_t(sig2), SR, frame_size=1024, hop_length=256)
    assert out_b.shape[0] == 2
    np.testing.assert_allclose(out_b[0].numpy(), out_0.numpy(), rtol=1e-10)
    np.testing.assert_allclose(out_b[1].numpy(), out_1.numpy(), rtol=1e-10)


# ---------------------------------------------------------------------------
# Differentiability
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('fn,kwargs', [
    (F_afx.mfcc, {'frame_size': 512, 'hop_length': 256}),
    (F_afx.melspectrogram, {'frame_size': 512, 'hop_length': 256}),
    (F_afx.spectral_centroid, {'frame_size': 512, 'hop_length': 256}),
    (F_afx.spectral_flatness, {'frame_size': 512, 'hop_length': 256}),
    (F_afx.rms_energy, {}),
])
def test_gradients_flow(signal, fn, kwargs):
    x = _t(signal[:4096]).clone().requires_grad_(True)
    out = fn(x, SR, **kwargs)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0  # gradient actually reaches the input


# ---------------------------------------------------------------------------
# AFXFeatureModule
# ---------------------------------------------------------------------------

def test_module_dict_output(signal):
    module = AFXFeatureModule(
        features={
            'mfcc': {'n_mfcc': 13, 'frame_size': 1024, 'hop_length': 256},
            'spectral_centroid': {'frame_size': 1024, 'hop_length': 256},
            'zcr': {'frame_size': 1024, 'hop_length': 256},
        },
        sample_rate=SR,
    )
    feats = module(_t(signal).to(torch.float32))
    assert set(feats) == {'mfcc', 'spectral_centroid', 'zcr'}
    assert feats['mfcc'].shape[0] == 13


def test_module_vector_output_batched(signal):
    module = AFXFeatureModule(
        features={
            'mfcc': {'n_mfcc': 13, 'frame_size': 1024, 'hop_length': 256},
            'spectral_centroid': {'frame_size': 1024, 'hop_length': 256},
            'rms_energy': {},
        },
        sample_rate=SR, aggregation='mean', output='vector',
    )
    batch = torch.stack([_t(signal), _t(signal[::-1].copy())]).to(torch.float32)
    vec = module(batch)
    assert vec.shape == (2, 13 + 1 + 1)


def test_module_matches_numpy_mean_aggregation(signal):
    """module(aggregation='mean') == extract_all_features + mean pooling."""
    from AFX.extract_all import extract_all_features
    config = {
        'sample_rate': SR,
        'features': {
            'mfcc': {'n_mfcc': 13, 'frame_size': 1024, 'hop_length': 256},
            'spectral_centroid': {'frame_size': 1024, 'hop_length': 256},
        },
        'aggregation': 'mean', 'flatten': True,
    }
    ref = extract_all_features(signal, SR, config)
    module = AFXFeatureModule(
        features=config['features'], sample_rate=SR, aggregation='mean',
    ).double()
    out = module(_t(signal))
    np.testing.assert_allclose(
        out['mfcc'].numpy().ravel(), ref['mfcc'].ravel(), rtol=1e-6
    )
    np.testing.assert_allclose(
        out['spectral_centroid'].numpy().ravel(),
        ref['spectral_centroid'].ravel(), rtol=1e-6,
    )


def test_module_is_trainable_frontend(signal):
    """Gradients flow through the module into upstream parameters."""
    x = _t(signal[:4096]).to(torch.float32)
    gain = torch.nn.Parameter(torch.tensor(1.0))
    module = AFXFeatureModule(
        features={'mfcc': {'n_mfcc': 13, 'frame_size': 512,
                           'hop_length': 256}},
        sample_rate=SR, aggregation='mean', output='vector',
    )
    loss = module(gain * x).sum()
    loss.backward()
    assert gain.grad is not None and torch.isfinite(gain.grad)


def test_module_rejects_unsupported_feature():
    with pytest.raises(ValueError, match='torch path'):
        AFXFeatureModule(features={'hnr': {}}, sample_rate=SR)


def test_supported_features_listed():
    names = supported_features()
    assert 'mfcc' in names and 'melspectrogram' in names


def test_module_device_movement(signal):
    """Buffers follow .to(); CUDA exercised only if available."""
    module = AFXFeatureModule(
        features={'mfcc': {'frame_size': 1024, 'hop_length': 256}},
        sample_rate=SR,
    )
    module = module.double()
    assert module._melfb_mfcc.dtype == torch.float64
    if torch.cuda.is_available():  # pragma: no cover
        module = module.to('cuda')
        out = module(_t(signal).cuda())
        assert out['mfcc'].is_cuda
