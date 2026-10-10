# Benchmarks

Wall time per feature on one 10 s clip at 22.05 kHz (a 220 Hz tone plus
white noise, float32), with the parameters of the bundled `config.json`.
Produced by

```bash
python benchmarks/bench_features.py --repeat 7 --librosa --torch-batch 16 --markdown
```

Each number is the median of 7 timed calls after one untimed warm-up; the
IQR column is the interquartile range of those 7 calls in ms. AFX's STFT
cache is cleared before every AFX call, so each feature pays for its own
transform. librosa runs with the same `n_fft`, `hop_length` and `center`
and with its default (disabled) cache. The torch column is one
`AFXFeatureModule` forward pass under `torch.no_grad()` on a (16, 220500)
float32 batch, divided by 16.

Machine: a 2 vCPU cloud VM, Intel Xeon Processor @ 2.10 GHz (model string
as reported by `lscpu`, 1 thread per core, 4 MiB L2 per core), Linux 6.18,
7 GB RAM. Python 3.13.16, afx-audio 0.3.0, NumPy 2.5.3, SciPy 1.18.1,
librosa 1.0.0, torch 2.14.1+cpu with 2 intra-op threads.

| Feature | AFX ms | IQR | librosa ms | librosa function | torch ms per clip (batch of 16) |
|---|---:|---:|---:|---|---:|
| `zcr` | 2.10 | 0.19 | 7.16 | `feature.zero_crossing_rate` | 4.78 |
| `variance` | 0.68 | 0.02 |  |  |  |
| `entropy` | 21.12 | 2.80 |  |  |  |
| `crest_factor` | 2.31 | 0.23 |  |  |  |
| `kurtosis` | 1.08 | 0.05 |  |  |  |
| `autocorr_variance` | 14.39 | 1.00 |  |  |  |
| `mobility` | 1.46 | 0.03 |  |  |  |
| `complexity` | 1.47 | 0.05 |  |  |  |
| `sample_entropy` | 320.44 | 35.74 |  |  |  |
| `energy_ratio` | 0.39 | 0.10 |  |  |  |
| `short_time_energy` | 0.37 | 0.03 |  |  |  |
| `rms_energy` | 0.36 | 0.03 |  |  | 0.07 |
| `spectral_centroid` | 10.86 | 0.16 | 10.80 | `feature.spectral_centroid` | 7.24 |
| `spectral_bandwidth` | 9.18 | 1.20 | 12.65 | `feature.spectral_bandwidth` | 8.78 |
| `spectral_rolloff` | 8.47 | 0.17 | 9.03 | `feature.spectral_rolloff` | 8.61 |
| `spectral_contrast` | 9.03 | 0.24 | 8.46 | `feature.spectral_contrast` | 20.51 |
| `spectral_entropy` | 7.81 | 0.19 |  |  | 7.23 |
| `spectral_flatness` | 8.08 | 1.55 | 6.71 | `feature.spectral_flatness` | 7.43 |
| `spectral_flux` | 7.21 | 0.19 |  |  | 5.66 |
| `spectral_skewness` | 10.96 | 0.89 |  |  | 10.82 |
| `spectral_slope` | 8.31 | 1.80 |  |  | 5.86 |
| `spectral_kurtosis` | 10.90 | 1.73 |  |  | 7.98 |
| `spectral_crest` | 7.08 | 0.59 |  |  | 4.91 |
| `spectral_decrease` | 7.77 | 0.07 |  |  | 3.44 |
| `melspectrogram` | 8.72 | 0.50 | 8.65 | `feature.melspectrogram` | 7.15 |
| `mfcc` | 11.08 | 0.43 | 14.67 | `feature.mfcc` | 4.27 |
| `mfcc_delta` | 9.49 | 0.44 | 10.00 | `feature.delta(mfcc)` | 4.34 |
| `mfcc_delta_delta` | 10.61 | 1.56 | 9.41 | `feature.delta(mfcc, order=2)` | 5.22 |
| `chroma_stft` | 8.75 | 0.40 | 7.27 | `feature.chroma_stft(tuning=0)` | 6.31 |
| `chroma_cqt` | 8.97 | 0.10 | 95.78 | `feature.chroma_cqt *` |  |
| `tonnetz` | 12.35 | 0.26 | 10.54 | `feature.tonnetz(chroma=chroma_stft)` |  |
| `onset_strength` | 10.73 | 2.35 | 10.24 | `onset.onset_strength` |  |
| `cqt` | 9.69 | 0.44 | 27.45 | `cqt *` |  |
| `pitch` | 21.98 | 2.82 | 25.23 | `yin(center=False) *` |  |
| `thd` | 4.47 | 0.75 |  |  |  |
| `hnr` | 20.49 | 0.86 |  |  |  |
| `lpc` | 12.57 | 0.72 |  |  |  |
| `formants` | 47.91 | 2.56 |  |  |  |
| `gfcc` | 8.58 | 0.13 |  |  | 4.29 |
| total | 678.21 |  |  |  | 142.51 (one module) |

`extract_all_features` on the same config, sharing one STFT per framing: 606.34 ms

## Reading the table

Rows marked `*` are not the same computation. AFX's `cqt` and `chroma_cqt`
weight STFT bins with log-spaced triangles, so they cost one STFT; librosa
computes a true constant-Q transform with octave-wise filtering and
resampling, which is 3 to 11 times slower here and has a time resolution
that varies with frequency. `pitch` and `librosa.yin` implement the same
estimator but differ in how unvoiced frames are reported (see
[librosa equivalence](librosa.md)).

For the rows that are numerically equal to librosa, the two libraries are
within about 35% of each other (`zcr` aside), because both are dominated
by one real FFT over about 430 frames of 2048 samples. AFX additionally
memoizes the mel and gammatone filterbanks across calls; librosa
rebuilds its mel filterbank on every call.

`sample_entropy` is about half of the total. Its cost grows with the
square of the frame length in the worst case (Richman and Moorman, 2000), and the
bundled config computes it on 215 frames of 1024 samples. Drop it, or use
shorter frames, when throughput matters.

`extract_all_features` reuses one STFT for features with the same
framing, centering and padding mode. With the bundled config that saves
about 70 ms of the 680 ms sum, since the features differ in `center` and
padding defaults; aligning those parameters in a config increases the
saving.

On this two-core machine the torch path runs most spectral features at
the same speed per clip as the NumPy path or up to 2.6 times faster
(MFCC 4.3 ms against 11.1 ms) by batching the FFT and matrix products.
`spectral_contrast` (a sort per octave band) and `zcr` are slower in
torch. Expect the gap to change with core count and on a GPU; rerun the
command above on the target machine.

## Variance

On a shared VM, single runs move by 10 to 30%: two consecutive runs of
the command above gave 8.8 and 11.1 ms for `mfcc` and 648 and 678 ms for
the total. The IQR column shows the spread within one run.
Compare libraries within one run of the script, not across runs or
machines.
