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
from AFX.torch import AFXFeatureModule, supported_features
from AFX.torch import functional as F_afx

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


@pytest.mark.parametrize('batched', [True, False])
def test_module_mean_std_shapes(signal, batched):
    """mean_std must add a trailing (mean, std) pair for per-frame features
    and concatenate [means, stds] for 2-D ones, for batched and unbatched
    input alike. It used to concatenate along the batch axis."""
    module = AFXFeatureModule(
        features={'spectral_centroid': {}, 'mfcc': {'n_mfcc': 4},
                  'rms_energy': {}},
        sample_rate=SR, aggregation='mean_std',
    ).double()
    x = _t(np.stack([signal, 0.5 * signal])) if batched else _t(signal)
    out = module(x)
    lead = (2,) if batched else ()
    assert tuple(out['spectral_centroid'].shape) == lead + (2,)
    assert tuple(out['mfcc'].shape) == lead + (8,)
    assert tuple(out['rms_energy'].shape) == lead + (2,)

    # Column order: means then stds, per sample
    centroid = F_afx.spectral_centroid(x, SR)
    expected = torch.stack(
        [centroid.mean(dim=-1), centroid.std(dim=-1, unbiased=False)], dim=-1
    )
    assert torch.allclose(out['spectral_centroid'], expected)

    vec = AFXFeatureModule(
        features={'spectral_centroid': {}, 'mfcc': {'n_mfcc': 4}},
        sample_rate=SR, aggregation='mean_std', output='vector',
    ).double()(x)
    assert tuple(vec.shape) == lead + (10,)


# ---------------------------------------------------------------------------
# Every torch-supported feature against its NumPy extractor
# ---------------------------------------------------------------------------

_MODULE_PARAMS = [
    ({}),
    ({'frame_size': 512, 'hop_length': 128, 'center': True}),
]


@pytest.mark.parametrize('params', _MODULE_PARAMS)
@pytest.mark.parametrize('name', supported_features())
def test_module_matches_numpy_extractor(signal, name, params):
    import AFX

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)  # build the constant buffers in float64
    try:
        module = AFXFeatureModule({name: params}, sample_rate=SR)
    finally:
        torch.set_default_dtype(default)
    out = module(_t(signal))[name].numpy()
    ref = AFX.FEATURE_MAP[name](signal, SR, **params)[name]
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, rtol=1e-6, atol=1e-6 * max(1.0, np.abs(ref).max()))


@pytest.mark.parametrize('name', ['spectral_contrast', 'chroma_stft', 'gfcc', 'mfcc_delta', 'spectral_skewness'])
def test_new_features_are_differentiable_and_batched(signal, name):
    module = AFXFeatureModule({name: {}}, sample_rate=SR, aggregation='mean').double()
    x = torch.stack([_t(signal), 0.5 * _t(signal[::-1].copy())]).requires_grad_(True)
    out = module(x)[name]
    assert out.shape[0] == 2
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    single = module(_t(signal))[name]
    torch.testing.assert_close(out[0].detach(), single)


@pytest.mark.parametrize('width, order', [(3, 1), (5, 2), (9, 1), (9, 2)])
def test_torch_delta_matches_numpy(width, order):
    from AFX.methods.delta import compute_delta

    feats = np.random.default_rng(3).standard_normal((4, 30))
    np.testing.assert_allclose(F_afx.delta(_t(feats), width, order=order).numpy(), compute_delta(feats, width, order),
                               atol=1e-12)


@pytest.mark.parametrize('name', ['spectral_bandwidth', 'spectral_skewness', 'spectral_flux', 'rms_energy',
                                  'spectral_contrast', 'mfcc_delta'])
def test_silent_frames_give_finite_gradients(name):
    x = np.random.default_rng(1).standard_normal(8000)
    x[:4000] = 0.0
    params = {'frame_size': 1024, 'hop_length': 512}
    module = AFXFeatureModule({name: params}, sample_rate=16000, aggregation='mean').double()
    wav = _t(x).requires_grad_(True)
    module(wav)[name].sum().backward()
    assert torch.isfinite(wav.grad).all()


def test_module_rejects_unsupported_settings():
    with pytest.raises(ValueError, match='order'):
        AFXFeatureModule({'mfcc_delta': {'order': 3}})
    with pytest.raises(ValueError, match='whole-clip'):
        AFXFeatureModule({'spectral_centroid': {'frame_size': None}})
    with pytest.raises(ValueError, match='order'):
        F_afx.delta(torch.zeros(2, 20), order=3)
    module = AFXFeatureModule(['mfcc', 'zcr'])
    assert module.feature_names() == ['mfcc', 'zcr']


def test_module_input_checks_and_summary_alias(signal):
    module = AFXFeatureModule(['zcr'], sample_rate=SR, aggregation='summary')
    assert module.aggregation == 'mean_std'
    with pytest.raises(TypeError, match='floating-point'):
        module(torch.zeros(1000, dtype=torch.int16))
    with pytest.raises(TypeError, match='torch.Tensor'):
        module(signal)


@pytest.mark.parametrize('aggregation, numpy_aggregation', [('mean_std', 'summary'), ('mean', 'mean')])
def test_module_column_names_match_numpy(signal, aggregation, numpy_aggregation):
    import AFX

    features = {'mfcc': {'n_mfcc': 4}, 'spectral_centroid': {}, 'zcr': {}}
    module = AFXFeatureModule(features, sample_rate=SR, aggregation=aggregation, output='vector').double()
    names = module.get_feature_names_out()
    ref = AFX.extract_all_features(signal, SR, {'features': features, 'aggregation': numpy_aggregation})
    ref_names, ref_values = AFX.flatten_features(ref)
    assert names == ref_names
    np.testing.assert_allclose(module(_t(signal)).numpy(), ref_values, rtol=1e-6, atol=1e-6)


def test_aggregation_skips_undefined_frames_like_numpy():
    import AFX

    x = np.concatenate([np.zeros(SR), np.random.default_rng(4).standard_normal(SR)])
    features = {'spectral_kurtosis': {}, 'rms_energy': {'frame_size': 1024, 'hop_length': 512}}
    module = AFXFeatureModule(features, sample_rate=SR, aggregation='mean_std', output='vector').double()
    wav = _t(x).requires_grad_(True)
    out = module(wav)
    ref = AFX.flatten_features(AFX.extract_all_features(x, SR, {'features': features, 'aggregation': 'summary'}))[1]
    np.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-6)
    out.sum().backward()
    assert torch.isfinite(wav.grad).all()
