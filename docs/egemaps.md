# eGeMAPS v02

AFX computes the extended Geneva Minimalistic Acoustic Parameter Set,
version 02: 25 low-level descriptors (LLDs) every 10 ms and 88 functionals
per clip. The implementation is written from the paper and from the
parameters of openSMILE's `eGeMAPSv02` configuration (frame lengths,
windows, band edges, filter counts, smoothing, functionals); it contains no
openSMILE code and is MIT licensed like the rest of AFX. openSMILE, whose
licence allows research use only, runs in `tests/test_egemaps.py` and
`benchmarks/validate_egemaps.py` to measure agreement and is not a
dependency.

## Usage

```python
import afx_audio as afx

signal, sr = afx.load_audio('speech.wav', sr=16000)
values = afx.egemaps(signal, sr)            # {name: value}, 88 entries
values['F0semitoneFrom27.5Hz_sma3nz_amean']

lld = afx.FEATURE_MAP['egemaps_lld'](signal, sr, return_metadata=True)
lld['egemaps_lld'].shape                    # (25, n_frames), rows afx.EGEMAPS_LLD_NAMES
```

Names and order are openSMILE's (`afx.EGEMAPS_NAMES`,
`afx.EGEMAPS_LLD_NAMES`), so tables computed with either tool line up
column by column. In configs, `egemaps` is a clip-level feature with one
column of 88 rows; mean aggregation passes the values through unchanged.
The `egemaps` preset sets this up for every front end:

```python
X = afx.AFXFeatureExtractor(config='egemaps').fit_transform(signals)   # (n, 88)
afx.extract_features_from_folder('clips/', afx.load_preset('egemaps'), 'egemaps.parquet')
```

or `afx clips/ --preset egemaps -o egemaps.parquet` on the command line.
The output columns are `egemaps_0` to `egemaps_87` in the order of
`afx.EGEMAPS_NAMES`.

The LLD array holds NaN where a descriptor is undefined: the 15 voice
descriptors in unvoiced frames, jitter and shimmer in frames with fewer than
two periods. openSMILE writes 0 there. Functionals over an empty set (for
example any voiced statistic of a clip without voicing) are 0, as in
openSMILE. Any sample rate from 8 kHz up works; 5 s of 16 kHz audio takes
0.1 to 0.2 s, a minute about 1.3 s with a peak of 150 MB.

## Descriptors

Frame i starts at i × 10 ms. Loudness, the spectral descriptors, MFCC and
formants use 20 ms Hamming frames; F0, jitter, shimmer, HNR and the
harmonic levels use 60 ms Gaussian frames (sigma 0.4). FFT lengths are the
next power of two.

| Descriptor | Computation |
|---|---|
| F0 (semitones above 27.5 Hz) | Subharmonic summation (Hermes, 1988) over 15 harmonics with compression 0.85 on a 48 points per octave log-frequency spectrum, 55 to 1000 Hz. A frame is voiced when its windowed RMS is at least 0.001 (full scale 1), the normalized cross-correlation at the period exceeds 0.3 and the SHS peak exceeds 8 times the SHS mean; gaps of up to 2 frames are bridged and voiced runs shorter than 3 frames dropped. Dynamic programming over the 6 best SHS candidates removes octave jumps, and the F0 is refined from the peaks of the first five harmonics. |
| Jitter, shimmer (dB) | Periods are tracked through each voiced stretch by waveform matching (next period within ±10 % of the F0 period, correlation at least 0.5). Per frame, over the periods starting in its 60 ms: mean absolute difference of consecutive period lengths over the mean length, and mean absolute dB difference of consecutive peak-to-peak amplitudes. |
| HNR (dB) | 10 log10(r / (1 - r)), r the largest autocorrelation within ±10 % of the F0 period over the zero-lag value, with the autocorrelation taken from the windowed power spectrum. |
| H1-H2, H1-A3, F1-F3 level | Harmonic levels h(k) are the largest magnitude within k F0 ± F0 / 2 in dB. Formant level i is h(k) - h(1) for the harmonic k closest to formant i. |
| F1-F3 frequency and bandwidth | The power spectrum up to 5.5 kHz, read as that of an 11 kHz signal, gives the autocorrelation for an order 11 LPC fit. Roots between 50 and 5450 Hz with a bandwidth below 1 kHz are the candidates (Markel and Gray, 1976). |
| Loudness | 26 triangular bands, equally spaced on the mel scale from 20 to 8000 Hz, on the power spectrum; each band is weighted by the equal-loudness curve of Hermansky (1990) and raised to the power 0.33; the bands are summed. |
| Alpha ratio, Hammarberg index | 10 log10 of the energy in 1-5 kHz over the energy below 1 kHz; 10 log10 of the strongest bin below 2 kHz over the strongest bin in 2-5 kHz. |
| Spectral slopes 0-500, 500-1500 Hz | Regression slope of the power spectrum in dB against frequency in Hz. |
| Spectral flux | RMS over the bins below 5 kHz of the change in magnitude spectrum from the previous frame. |
| MFCC 1-4 | HTK style: the 26 mel band energies, natural log, DCT-II, sinusoidal lifter 22. |
| Equivalent sound level | 10 log10 of the mean power of the 20 ms frames (unwindowed), floored at -120 dB. |

Every LLD is smoothed by a 3-frame moving average; the voice descriptors
(the `sma3nz` rows) only over voiced neighbours. The spectral descriptors
are also gated into voiced (`...V`) and unvoiced (`...UV`) versions before
smoothing.

The functionals are the arithmetic mean (`amean`), the standard deviation
over the mean (`stddevNorm`), the 20th, 50th and 80th percentiles with
linear interpolation and their 20-80 range, and the mean and standard
deviation of the rising and falling slopes between peaks and valleys of the
F0 and loudness contours. A peak needs a rise of more than 10 % of the
contour's range from the preceding valley. Statistics of voice descriptors
only use frames where the descriptor is defined; F0 contours of separate
voiced stretches are joined before the slopes are taken, as in openSMILE.

## Where AFX follows openSMILE rather than the paper's text

The paper describes some parameters differently from what openSMILE
computes. Published eGeMAPS numbers come from openSMILE, so AFX matches its
output in these cases.

| Item | Paper | openSMILE and AFX |
|---|---|---|
| H1-H2, H1-A3 | First and second F0 harmonic | Harmonics are counted from 2 F0: 'H1-H2' is h(2) - h(3) and 'H1-A3' is h(2) - A3. Formant levels are relative to h(1). |
| Spectral slope | Linear regression | (N' sum f y - sum f sum y) / (N' sum f² - (sum f)²) with N' = N - 1 for N bins, which is not the least-squares slope. |
| Alpha ratio | 50-1000 Hz against 1-5 kHz | The low band starts at 0 Hz. |
| Segment lengths | Length of voiced and unvoiced regions | (frames - 1) × 10 ms. |
| Falling slope | Rate of fall | Reported as a positive rate, except a fall that ends the contour, which keeps its negative sign. |
| Loudness | Perceived intensity from an auditory spectrum | Equal-loudness-weighted, compressed mel band sum (PLP auditory spectrum). AFX replaces Hermansky's high-frequency term with a roll-off 1 / (1 + (f / 4580 Hz)^6.88) and a gain of 0.069, fitted to openSMILE's loudness of pure tones from 40 Hz to 7.9 kHz; with Hermansky's term tones above 4 kHz are off by more than 20 %. |

## Where AFX differs from openSMILE

| Item | openSMILE | AFX |
|---|---|---|
| Formants | No bandwidth limit, so broad poles from the spectral tilt are often reported as F1 or F2 (bandwidths of 1 to 1.4 kHz are typical) | Candidates with a bandwidth of 1 kHz or more are discarded; see the accuracy table below |
| Formant levels in unvoiced frames | A -201 dB placeholder that enters the `F*amplitudeLogRelF0` functionals, so their means mostly measure the unvoiced fraction | Voiced frames only, like every other voice descriptor |
| F0 and voicing | SHS with a Viterbi smoother; the exact rules are not documented | SHS candidates, a voicing rule tuned to agree with openSMILE (91.5 % of frames on the LibriSpeech segments, 98.8 % on the synthetic corpus), dynamic programming, harmonic-peak refinement. Where both mark a frame voiced, the F0 differs by 0.04 (synthetic) and 0.07 (speech) semitones in the median. Pure tones from 60 Hz to 880 Hz come out within 0.03 semitones |
| Jitter and shimmer | Period search by cross-correlation, details not documented | Own waveform-matching period tracker (above) |
| Peaks and slopes | Undocumented peak rules with odd edge cases (in one test a contour without a peak reported its rising slope per frame instead of per second) | The hysteresis rule above; fed with openSMILE's own F0 and loudness contours it reproduces openSMILE's slope statistics with r = 0.97 to 0.998 |
| Clip ends | The functionals see 1 to 3 frames more or fewer than the LLD output | 1 + (N - 20 ms) / 10 ms frames for the spectral descriptors, 1 + (N - 60 ms) / 10 ms for the voice descriptors |

## Agreement with openSMILE

Pearson r across clips and median relative error |AFX - openSMILE| /
|openSMILE| for every functional, from `python
benchmarks/validate_egemaps.py --real --write-docs` (openSMILE 2.6.0). The
synthetic corpus is 20 clips from `tests/voice_synth.py`: Rosenberg glottal
pulses with arch, glide or vibrato F0 contours (90 to 380 Hz), jitter up to
3 %, shimmer up to 15 %, five vowels, 1 to 3 voiced stretches with
fricative noise or silence between them, SNR 5 to 40 dB. The speech corpus
is the three LibriSpeech example clips shipped with librosa, cut into 13
segments of 3 s. A correlation is only meaningful when the functional
varies across the corpus; the relative error is the better guide where the
spread is small (for example `F0..._stddevNorm` on speech).

Loudness, spectral flux, MFCC, equivalent sound level and the F0 level
statistics (mean and percentiles) agree within 3 % on both corpora (r of
0.99 or more). On the synthetic corpus the voiced and unvoiced spectral
means, HNR, H1-H2 and the segment statistics agree within 2 % as well (the
unvoiced Hammarberg index within 8 %). On speech the spectral means
correlate at 0.86 to 0.99 and differ by up to 22 %, the segment statistics
by up to 30 %, because the two voicing decisions differ in about 8 % of the
frames, mostly at the edges of voiced stretches. Frame by frame, where both
mark a frame voiced, HNR correlates at 0.99 and H1-H2 at 0.98 on speech;
the clip means of H1-H2 still differ because openSMILE reports values near
-120 dB in frames where it tracks F0 near 55 Hz. F0 slopes, jitter and
shimmer mostly correlate (r of 0.5 to 0.99, except the spread of falling F0
slopes) but differ by 10 to 50 %: they depend on frame-by-frame voicing and
period decisions that are not documented. The formant frequencies,
bandwidths and levels do not agree, by design (previous section).

<!-- agreement table start -->
| Functional | r (synthetic) | rel. error (synthetic) | r (speech) | rel. error (speech) |
|---|---:|---:|---:|---:|
| `F0semitoneFrom27.5Hz_sma3nz_amean` | 1.000 | 0.1 % | 0.997 | 0.8 % |
| `F0semitoneFrom27.5Hz_sma3nz_stddevNorm` | 0.995 | 1.8 % | 0.741 | 6.5 % |
| `F0semitoneFrom27.5Hz_sma3nz_percentile20.0` | 0.999 | 0.1 % | 0.999 | 0.3 % |
| `F0semitoneFrom27.5Hz_sma3nz_percentile50.0` | 1.000 | 0.1 % | 1.000 | 0.6 % |
| `F0semitoneFrom27.5Hz_sma3nz_percentile80.0` | 1.000 | 0.2 % | 0.993 | 0.7 % |
| `F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2` | 0.989 | 2.1 % | 0.994 | 5.0 % |
| `F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope` | 0.971 | 24.4 % | 0.877 | 34.9 % |
| `F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope` | 0.936 | 2.6 % | 0.908 | 43.7 % |
| `F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope` | 0.998 | 26.5 % | 0.805 | 49.2 % |
| `F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope` | 0.141 | 10.6 % | 0.626 | 76.1 % |
| `loudness_sma3_amean` | 1.000 | 0.1 % | 1.000 | 0.1 % |
| `loudness_sma3_stddevNorm` | 1.000 | 0.2 % | 1.000 | 0.0 % |
| `loudness_sma3_percentile20.0` | 1.000 | 0.3 % | 1.000 | 0.1 % |
| `loudness_sma3_percentile50.0` | 1.000 | 0.1 % | 1.000 | 0.1 % |
| `loudness_sma3_percentile80.0` | 1.000 | 0.1 % | 1.000 | 0.1 % |
| `loudness_sma3_pctlrange0-2` | 1.000 | 0.2 % | 1.000 | 0.1 % |
| `loudness_sma3_meanRisingSlope` | 0.951 | 15.4 % | 0.987 | 2.0 % |
| `loudness_sma3_stddevRisingSlope` | 0.852 | 21.5 % | 0.862 | 5.8 % |
| `loudness_sma3_meanFallingSlope` | 0.348 | 7.5 % | 0.977 | 2.5 % |
| `loudness_sma3_stddevFallingSlope` | 0.981 | 5.1 % | 0.978 | 2.6 % |
| `spectralFlux_sma3_amean` | 1.000 | 0.1 % | 1.000 | 0.0 % |
| `spectralFlux_sma3_stddevNorm` | 1.000 | 0.1 % | 1.000 | 0.0 % |
| `mfcc1_sma3_amean` | 1.000 | 0.8 % | 1.000 | 0.3 % |
| `mfcc1_sma3_stddevNorm` | 1.000 | 0.9 % | 1.000 | 0.8 % |
| `mfcc2_sma3_amean` | 1.000 | 1.5 % | 1.000 | 2.7 % |
| `mfcc2_sma3_stddevNorm` | 0.996 | 2.0 % | 0.996 | 2.6 % |
| `mfcc3_sma3_amean` | 1.000 | 1.0 % | 1.000 | 1.1 % |
| `mfcc3_sma3_stddevNorm` | 0.995 | 1.5 % | 1.000 | 1.0 % |
| `mfcc4_sma3_amean` | 1.000 | 1.7 % | 1.000 | 1.9 % |
| `mfcc4_sma3_stddevNorm` | 0.998 | 2.3 % | 0.958 | 2.2 % |
| `jitterLocal_sma3nz_amean` | 0.919 | 10.7 % | 0.513 | 16.6 % |
| `jitterLocal_sma3nz_stddevNorm` | 0.686 | 42.5 % | 0.444 | 41.6 % |
| `shimmerLocaldB_sma3nz_amean` | 0.765 | 14.7 % | 0.544 | 22.2 % |
| `shimmerLocaldB_sma3nz_stddevNorm` | 0.524 | 23.5 % | 0.633 | 9.1 % |
| `HNRdBACF_sma3nz_amean` | 0.989 | 1.2 % | 0.940 | 10.6 % |
| `HNRdBACF_sma3nz_stddevNorm` | 0.491 | 11.7 % | 0.632 | 41.8 % |
| `logRelF0-H1-H2_sma3nz_amean` | 1.000 | 0.3 % | 0.464 | 7.2 % |
| `logRelF0-H1-H2_sma3nz_stddevNorm` | 0.999 | 3.2 % | 0.294 | 17.7 % |
| `logRelF0-H1-A3_sma3nz_amean` | 0.913 | 19.0 % | 0.879 | 9.7 % |
| `logRelF0-H1-A3_sma3nz_stddevNorm` | 0.270 | 38.6 % | -0.529 | 45.2 % |
| `F1frequency_sma3nz_amean` | 0.189 | 24.8 % | 0.572 | 36.7 % |
| `F1frequency_sma3nz_stddevNorm` | -0.581 | 87.6 % | -0.513 | 70.8 % |
| `F1bandwidth_sma3nz_amean` | 0.188 | 92.6 % | 0.437 | 84.7 % |
| `F1bandwidth_sma3nz_stddevNorm` | -0.033 | 396.1 % | -0.169 | 240.1 % |
| `F1amplitudeLogRelF0_sma3nz_amean` | 0.162 | 109.4 % | 0.034 | 93.9 % |
| `F1amplitudeLogRelF0_sma3nz_stddevNorm` | 0.241 | 123.9 % | -0.214 | 26.4 % |
| `F2frequency_sma3nz_amean` | -0.824 | 42.0 % | 0.756 | 12.8 % |
| `F2frequency_sma3nz_stddevNorm` | 0.245 | 70.2 % | -0.544 | 106.0 % |
| `F2bandwidth_sma3nz_amean` | -0.056 | 86.8 % | 0.219 | 66.2 % |
| `F2bandwidth_sma3nz_stddevNorm` | 0.202 | 165.3 % | -0.284 | 65.6 % |
| `F2amplitudeLogRelF0_sma3nz_amean` | 0.130 | 102.3 % | 0.058 | 76.1 % |
| `F2amplitudeLogRelF0_sma3nz_stddevNorm` | -0.024 | 120.4 % | 0.201 | 48.2 % |
| `F3frequency_sma3nz_amean` | -0.676 | 15.3 % | 0.101 | 9.6 % |
| `F3frequency_sma3nz_stddevNorm` | -0.155 | 50.7 % | 0.088 | 80.5 % |
| `F3bandwidth_sma3nz_amean` | 0.286 | 57.2 % | -0.092 | 60.6 % |
| `F3bandwidth_sma3nz_stddevNorm` | 0.699 | 26.6 % | 0.070 | 44.5 % |
| `F3amplitudeLogRelF0_sma3nz_amean` | 0.326 | 82.2 % | 0.170 | 68.2 % |
| `F3amplitudeLogRelF0_sma3nz_stddevNorm` | 0.059 | 80.8 % | 0.243 | 60.8 % |
| `alphaRatioV_sma3nz_amean` | 0.999 | 1.7 % | 0.968 | 3.9 % |
| `alphaRatioV_sma3nz_stddevNorm` | 0.962 | 9.0 % | 0.836 | 6.7 % |
| `hammarbergIndexV_sma3nz_amean` | 0.999 | 0.6 % | 0.986 | 1.2 % |
| `hammarbergIndexV_sma3nz_stddevNorm` | 0.959 | 5.1 % | 0.939 | 4.7 % |
| `slopeV0-500_sma3nz_amean` | 0.999 | 0.5 % | 0.998 | 6.2 % |
| `slopeV0-500_sma3nz_stddevNorm` | 0.987 | 4.0 % | 0.981 | 6.4 % |
| `slopeV500-1500_sma3nz_amean` | 0.999 | 1.4 % | 0.990 | 12.9 % |
| `slopeV500-1500_sma3nz_stddevNorm` | 0.988 | 9.6 % | 0.926 | 11.2 % |
| `spectralFluxV_sma3nz_amean` | 1.000 | 0.8 % | 0.998 | 5.0 % |
| `spectralFluxV_sma3nz_stddevNorm` | 0.965 | 2.1 % | 0.984 | 1.8 % |
| `mfcc1V_sma3nz_amean` | 0.997 | 1.3 % | 0.992 | 2.3 % |
| `mfcc1V_sma3nz_stddevNorm` | 0.817 | 12.9 % | 0.900 | 4.9 % |
| `mfcc2V_sma3nz_amean` | 1.000 | 1.1 % | 0.990 | 14.4 % |
| `mfcc2V_sma3nz_stddevNorm` | 0.999 | 2.5 % | 0.650 | 19.0 % |
| `mfcc3V_sma3nz_amean` | 1.000 | 0.9 % | 0.995 | 9.0 % |
| `mfcc3V_sma3nz_stddevNorm` | 0.995 | 5.2 % | 0.961 | 7.5 % |
| `mfcc4V_sma3nz_amean` | 1.000 | 0.6 % | 0.998 | 7.5 % |
| `mfcc4V_sma3nz_stddevNorm` | 1.000 | 4.9 % | 0.732 | 8.8 % |
| `alphaRatioUV_sma3nz_amean` | 0.998 | 1.1 % | 0.857 | 21.5 % |
| `hammarbergIndexUV_sma3nz_amean` | 0.998 | 8.2 % | 0.886 | 7.6 % |
| `slopeUV0-500_sma3nz_amean` | 0.998 | 0.5 % | 0.980 | 14.4 % |
| `slopeUV500-1500_sma3nz_amean` | 0.998 | 1.1 % | 0.943 | 19.6 % |
| `spectralFluxUV_sma3nz_amean` | 0.999 | 2.1 % | 0.964 | 13.7 % |
| `loudnessPeaksPerSec` | 0.978 | 0.0 % | 1.000 | 0.0 % |
| `VoicedSegmentsPerSec` | 1.000 | 0.2 % | 0.879 | 13.1 % |
| `MeanVoicedSegmentLengthSec` | 1.000 | 1.4 % | 0.941 | 29.3 % |
| `StddevVoicedSegmentLengthSec` | 1.000 | 0.0 % | 0.946 | 6.8 % |
| `MeanUnvoicedSegmentLength` | 1.000 | 1.7 % | 0.926 | 10.5 % |
| `StddevUnvoicedSegmentLength` | 1.000 | 1.3 % | 0.970 | 24.2 % |
| `equivalentSoundLevel_dBp` | 1.000 | 0.0 % | 1.000 | 0.0 % |
<!-- agreement table end -->

## Formant accuracy

Mean F1-F3 frequency of each synthetic clip against the formant
frequencies it was synthesized with (same run as above).

<!-- formant truth table start -->
| Formant | AFX r | AFX mean abs. error | openSMILE r | openSMILE mean abs. error |
|---|---:|---:|---:|---:|
| F1 | 0.992 | 18 Hz | 0.217 | 245 Hz |
| F2 | 0.998 | 38 Hz | -0.825 | 803 Hz |
| F3 | 0.945 | 86 Hz | -0.642 | 483 Hz |
<!-- formant truth table end -->

## Reference

F. Eyben, K. R. Scherer, B. W. Schuller, J. Sundberg, E. André, C. Busso,
L. Y. Devillers, J. Epps, P. Laukka, S. S. Narayanan and K. P. Truong,
"The Geneva Minimalistic Acoustic Parameter Set (GeMAPS) for Voice Research
and Affective Computing", IEEE Transactions on Affective Computing 7(2),
190-202, 2016. doi:10.1109/TAFFC.2015.2457417

```bibtex
@article{eyben2016gemaps,
  author  = {Eyben, Florian and Scherer, Klaus R. and Schuller, Bj{\"o}rn W. and Sundberg, Johan and
             Andr{\'e}, Elisabeth and Busso, Carlos and Devillers, Laurence Y. and Epps, Julien and
             Laukka, Petri and Narayanan, Shrikanth S. and Truong, Khiet P.},
  title   = {The {Geneva} Minimalistic Acoustic Parameter Set ({GeMAPS}) for Voice Research and Affective Computing},
  journal = {IEEE Transactions on Affective Computing},
  volume  = {7},
  number  = {2},
  pages   = {190--202},
  year    = {2016},
  doi     = {10.1109/TAFFC.2015.2457417}
}
```

D. J. Hermes, "Measurement of pitch by subharmonic summation", JASA 83(1),
257-264, 1988. H. Hermansky, "Perceptual linear predictive (PLP) analysis
of speech", JASA 87(4), 1738-1752, 1990. J. D. Markel and A. H. Gray,
Linear Prediction of Speech, Springer, 1976.
