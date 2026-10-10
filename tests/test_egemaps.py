"""eGeMAPS v02: shapes, names, synthetic sanity checks and agreement with openSMILE."""
import time

import numpy as np
import pytest

import afx_audio
from afx_audio.extractors.egemaps import extract_egemaps, extract_egemaps_lld
from tests.voice_synth import VOWELS, corpus, synthetic_voice

SR = 16000
NAMES = list(afx_audio.EGEMAPS_NAMES)
LLD = list(afx_audio.EGEMAPS_LLD_NAMES)


def periodic(periods, amplitudes):
    """A fixed 80-sample glottal pulse repeated at the given period lengths (samples) and amplitudes."""
    from tests.voice_synth import rosenberg

    pulse = np.diff(rosenberg(np.arange(81) / 80))
    out = []
    for n, a in zip(periods, amplitudes, strict=True):
        period = np.zeros(n)
        period[:80] = a * pulse
        out.append(period)
    return np.concatenate(out)


@pytest.fixture(scope='module')
def voice():
    return synthetic_voice(sr=SR, duration=3.0)


def test_names():
    assert len(NAMES) == 88 and len(set(NAMES)) == 88
    assert len(LLD) == 25 and len(set(LLD)) == 25
    assert NAMES[0] == 'F0semitoneFrom27.5Hz_sma3nz_amean'
    assert NAMES[-1] == 'equivalentSoundLevel_dBp'


def test_lld_shape_times_and_nan_where_unvoiced(voice):
    out = extract_egemaps_lld(voice, SR, return_metadata=True)
    values = out['egemaps_lld']
    n_frames = 1 + (len(voice) - 320) // 160
    assert values.shape == (25, n_frames)
    assert out['metadata']['names'] == LLD
    np.testing.assert_allclose(out['metadata']['times'][:2], [0.01, 0.02])
    assert np.isfinite(values[:10]).all()
    unvoiced = np.isnan(values[10])
    assert 0.1 < unvoiced.mean() < 0.5
    # every voice-quality row is undefined at least where F0 is
    assert np.isnan(values[11:, unvoiced]).all()


def test_functionals_vector_and_dict_agree(voice):
    out = extract_egemaps(voice, SR, return_metadata=True)
    assert out['egemaps'].shape == (88, 1)
    assert out['metadata']['names'] == NAMES
    as_dict = afx_audio.egemaps(voice, SR)
    assert list(as_dict) == NAMES
    np.testing.assert_allclose(list(as_dict.values()), out['egemaps'][:, 0])
    assert np.isfinite(out['egemaps']).all()


def test_config_integration(voice):
    feats = afx_audio.extract_all_features(voice, SR, {'features': ['egemaps', 'mfcc']})
    np.testing.assert_allclose(feats['egemaps'], extract_egemaps(voice, SR)['egemaps'][:, 0])
    names, values = afx_audio.flatten_features(feats)
    assert names[:2] == ['egemaps_0', 'egemaps_1'] and len(values) == 88 + 13
    info = afx_audio.feature_info('egemaps')
    assert info['rows'] == '88' and info['parameters'] == {}


def test_preset_and_sklearn_front_end(voice):
    expected = extract_egemaps(voice, SR)['egemaps'][:, 0]
    feats = afx_audio.extract_all_features(voice, SR, afx_audio.load_preset('egemaps'))
    np.testing.assert_allclose(feats['egemaps'], expected)
    pytest.importorskip('sklearn')
    from afx_audio.sklearn import AFXFeatureExtractor

    X = AFXFeatureExtractor(config='egemaps').fit_transform([voice, 0.5 * voice])
    assert X.shape == (2, 88)
    np.testing.assert_allclose(X[0], expected)


@pytest.mark.parametrize('f0', [110.0, 220.0, 440.0])
def test_pure_tone_f0(f0):
    t = np.arange(2 * SR) / SR
    values = afx_audio.egemaps(0.3 * np.sin(2 * np.pi * f0 * t), SR)
    assert values['F0semitoneFrom27.5Hz_sma3nz_amean'] == pytest.approx(12 * np.log2(f0 / 27.5), abs=0.1)
    assert values['F0semitoneFrom27.5Hz_sma3nz_stddevNorm'] < 1e-3
    assert values['VoicedSegmentsPerSec'] == pytest.approx(1 / 1.95, rel=0.05)


def test_periodic_signal_has_no_jitter_or_shimmer():
    x = 0.3 * periodic([128] * 250, [1.0] * 250)
    values = afx_audio.egemaps(x, SR)
    assert values['F0semitoneFrom27.5Hz_sma3nz_amean'] == pytest.approx(12 * np.log2(125 / 27.5), abs=0.05)
    assert values['jitterLocal_sma3nz_amean'] < 1e-3
    assert values['shimmerLocaldB_sma3nz_amean'] < 0.01


def test_known_jitter_and_shimmer():
    # alternating periods of 126 and 130 samples: |dT| / mean T = 4 / 128
    x = 0.3 * periodic([126, 130] * 125, [1.0] * 250)
    assert afx_audio.egemaps(x, SR)['jitterLocal_sma3nz_amean'] == pytest.approx(4 / 128, rel=0.15)
    # alternating amplitudes 1 and 0.8: |20 log10(0.8)| = 1.94 dB
    x = 0.3 * periodic([128] * 250, [1.0, 0.8] * 125)
    assert afx_audio.egemaps(x, SR)['shimmerLocaldB_sma3nz_amean'] == pytest.approx(20 * np.log10(1.25), rel=0.1)


def steady_voice(vowel, f0, jitter, shimmer, seed=3):
    """1.5 s of a constant-F0 vowel at 40 dB SNR with unambiguous periods and amplitudes, and its realized values."""
    return synthetic_voice(duration=2.0, f0=(f0, f0), contour='glide', jitter=jitter, shimmer=shimmer,
                           formants=VOWELS[vowel][0], bandwidths=VOWELS[vowel][1], snr_db=40.0,
                           voiced=((0.25, 1.75),), fricative_db=None, fixed_pulse=True, seed=seed, return_truth=True)


@pytest.mark.parametrize('vowel, f0', [('a', 140.0), ('u', 120.0), ('a', 220.0), ('i', 240.0)])
def test_random_jitter_matches_the_synthesis(vowel, f0):
    x, truth = steady_voice(vowel, f0, 0.02, 0.1)
    assert afx_audio.egemaps(x, SR)['jitterLocal_sma3nz_amean'] == pytest.approx(truth['jitter'], rel=0.1)
    x, _ = steady_voice(vowel, f0, 0.0, 0.0)
    assert afx_audio.egemaps(x, SR)['jitterLocal_sma3nz_amean'] < 0.0025


# /i/ near 240 Hz is left out: F1 (270 Hz) sits on the first harmonic and shimmer reads up to 40 % high there
@pytest.mark.parametrize('vowel, f0', [('a', 140.0), ('u', 120.0), ('a', 220.0)])
def test_random_shimmer_matches_the_synthesis(vowel, f0):
    x, truth = steady_voice(vowel, f0, 0.02, 0.1)
    assert afx_audio.egemaps(x, SR)['shimmerLocaldB_sma3nz_amean'] == pytest.approx(truth['shimmer_db'], rel=0.15)
    x, _ = steady_voice(vowel, f0, 0.0, 0.0)
    assert afx_audio.egemaps(x, SR)['shimmerLocaldB_sma3nz_amean'] < 0.25


@pytest.mark.slow
def test_jitter_and_shimmer_track_the_synthesis_across_voices():
    true, measured = [], []
    for v, vowel in enumerate('aiu'):
        for q, f0 in enumerate((120.0, 170.0, 240.0)):
            for k, jitter in enumerate((0.0, 0.01, 0.02, 0.03)):
                shimmer = (0.0, 0.05, 0.1, 0.15)[(k + v + q) % 4]
                x, truth = steady_voice(vowel, f0, jitter, shimmer, seed=10 * k + v)
                values = afx_audio.egemaps(x, SR)
                true.append([truth['jitter'], truth['shimmer_db']])
                measured.append([values['jitterLocal_sma3nz_amean'], values['shimmerLocaldB_sma3nz_amean']])
    true, measured = np.array(true), np.array(measured)
    for j, floor in ((0, 0.004), (1, 0.35)):
        slope = np.polyfit(true[:, j], measured[:, j], 1)[0]
        assert 0.85 < slope < 1.15
        assert np.corrcoef(true[:, j], measured[:, j])[0, 1] > 0.95
        assert measured[true[:, j] < 1e-9, j].max() < floor


def test_hnr_increases_with_snr():
    hnr = [afx_audio.egemaps(synthetic_voice(sr=SR, snr_db=snr, jitter=0.002, fricative_db=None, seed=1), SR)
           ['HNRdBACF_sma3nz_amean'] for snr in (0, 10, 20, 40)]
    assert np.all(np.diff(hnr) > 0), hnr


def test_formants_match_the_synthesis_filters():
    errors = []
    for _, x, params in corpus(10):
        values = afx_audio.egemaps(x, SR)
        errors.append([values[f'F{j}frequency_sma3nz_amean'] - params['formants'][j - 1] for j in (1, 2)])
    assert np.mean(np.abs(errors), axis=0) == pytest.approx([0, 0], abs=60)


def test_silence_and_short_input():
    for x in (np.zeros(SR), np.zeros(37), np.ones(1)):
        values = extract_egemaps(x, SR)['egemaps']
        assert np.isfinite(values).all()
        lld = extract_egemaps_lld(x, SR)['egemaps_lld']
        assert lld.shape[0] == 25 and np.isnan(lld[10:]).all()
    assert afx_audio.egemaps(np.zeros(SR), SR)['equivalentSoundLevel_dBp'] == pytest.approx(-120)


def test_level_follows_equivalent_sound_level(voice):
    quiet, loud = afx_audio.egemaps(0.1 * voice, SR), afx_audio.egemaps(voice, SR)
    assert loud['equivalentSoundLevel_dBp'] - quiet['equivalentSoundLevel_dBp'] == pytest.approx(20, abs=1e-6)
    assert loud['F0semitoneFrom27.5Hz_sma3nz_amean'] == pytest.approx(quiet['F0semitoneFrom27.5Hz_sma3nz_amean'],
                                                                      abs=0.05)


def test_other_sample_rate():
    t = np.arange(2 * 44100) / 44100
    values = afx_audio.egemaps(0.3 * np.sin(2 * np.pi * 200 * t), 44100)
    assert values['F0semitoneFrom27.5Hz_sma3nz_amean'] == pytest.approx(12 * np.log2(200 / 27.5), abs=0.1)
    with pytest.raises(ValueError, match='8000'):
        afx_audio.egemaps(np.zeros(4000), 4000)


def test_five_seconds_take_well_under_a_second():
    x = synthetic_voice(sr=SR, duration=5.0, voiced=((0.2, 2.3), (2.6, 4.8)))
    afx_audio.egemaps(x, SR)
    start = time.perf_counter()
    afx_audio.egemaps(x, SR)
    assert time.perf_counter() - start < 1.0


# Functionals that track openSMILE with r > 0.9 over the synthetic corpus (docs/egemaps.md has all 88)
CLOSE = [
    'F0semitoneFrom27.5Hz_sma3nz_amean', 'F0semitoneFrom27.5Hz_sma3nz_percentile50.0',
    'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2', 'loudness_sma3_amean', 'loudness_sma3_stddevNorm',
    'loudness_sma3_percentile80.0', 'spectralFlux_sma3_amean', 'mfcc1_sma3_amean', 'mfcc2_sma3_amean',
    'mfcc3_sma3_amean', 'mfcc4_sma3_amean', 'HNRdBACF_sma3nz_amean',
    'logRelF0-H1-H2_sma3nz_amean', 'alphaRatioV_sma3nz_amean', 'hammarbergIndexV_sma3nz_amean',
    'slopeV0-500_sma3nz_amean', 'slopeV500-1500_sma3nz_amean', 'spectralFluxV_sma3nz_amean',
    'mfcc1V_sma3nz_amean', 'alphaRatioUV_sma3nz_amean', 'hammarbergIndexUV_sma3nz_amean',
    'loudnessPeaksPerSec', 'VoicedSegmentsPerSec', 'MeanVoicedSegmentLengthSec', 'MeanUnvoicedSegmentLength',
    'equivalentSoundLevel_dBp',
]


@pytest.fixture(scope='module')
def opensmile_pairs():
    opensmile = pytest.importorskip('opensmile')
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.Functionals)
    clips = corpus(20)
    assert list(smile.process_signal(clips[0][1], SR).columns) == NAMES
    ours = np.array([extract_egemaps(x, SR)['egemaps'][:, 0] for _, x, _ in clips])
    ref = np.array([smile.process_signal(x, SR)[NAMES].to_numpy()[0] for _, x, _ in clips], dtype=float)
    return ours, ref


@pytest.mark.slow
@pytest.mark.parametrize('name', CLOSE)
def test_agreement_with_opensmile(opensmile_pairs, name):
    ours, ref = opensmile_pairs
    i = NAMES.index(name)
    assert np.corrcoef(ours[:, i], ref[:, i])[0, 1] > 0.9


@pytest.mark.slow
@pytest.mark.parametrize('name', ['loudness_sma3_amean', 'spectralFlux_sma3_amean', 'mfcc1_sma3_amean',
                                  'HNRdBACF_sma3nz_amean', 'logRelF0-H1-H2_sma3nz_amean',
                                  'F0semitoneFrom27.5Hz_sma3nz_amean', 'equivalentSoundLevel_dBp'])
def test_values_close_to_opensmile(opensmile_pairs, name):
    ours, ref = opensmile_pairs
    i = NAMES.index(name)
    assert np.median(np.abs(ours[:, i] - ref[:, i]) / np.abs(ref[:, i])) < 0.02


@pytest.mark.slow
def test_spectral_llds_match_opensmile_frame_by_frame():
    opensmile = pytest.importorskip('opensmile')
    smile = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                            feature_level=opensmile.FeatureLevel.LowLevelDescriptors)
    x = synthetic_voice(sr=SR, duration=3.0)
    ref = smile.process_signal(x, SR)
    ours = extract_egemaps_lld(x, SR)['egemaps_lld']
    n = len(ref) - 2
    for row in range(10):
        a, b = ours[row, 2:n], ref[LLD[row]].to_numpy()[2:n]
        assert np.corrcoef(a, b)[0, 1] > 0.995, LLD[row]


@pytest.mark.slow
def test_jitter_and_shimmer_closer_to_the_synthesis_than_opensmile(opensmile_pairs):
    # openSMILE reads jitter low at high F0 and shimmer flat, so agreement with it is not the target here
    ours, ref = opensmile_pairs
    truth = np.array([[p['truth']['jitter'], p['truth']['shimmer_db']] for _, _, p in corpus(20)])
    for j, name in enumerate(('jitterLocal_sma3nz_amean', 'shimmerLocaldB_sma3nz_amean')):
        i = NAMES.index(name)
        assert np.corrcoef(ours[:, i], ref[:, i])[0, 1] > 0.7
        assert np.mean(np.abs(ours[:, i] - truth[:, j])) < np.mean(np.abs(ref[:, i] - truth[:, j]))
