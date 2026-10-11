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
| `zcr` | 2.14 | 0.31 | 5.25 | `feature.zero_crossing_rate` | 5.74 |
| `variance` | 0.47 | 0.13 |  |  |  |
| `entropy` | 16.60 | 3.46 |  |  |  |
| `crest_factor` | 1.84 | 0.06 |  |  |  |
| `kurtosis` | 0.76 | 0.04 |  |  |  |
| `autocorr_variance` | 9.09 | 0.66 |  |  |  |
| `mobility` | 1.10 | 0.02 |  |  |  |
| `complexity` | 1.36 | 0.09 |  |  |  |
| `sample_entropy` | 267.52 | 4.15 |  |  |  |
| `energy_ratio` | 0.33 | 0.06 |  |  |  |
| `short_time_energy` | 0.36 | 0.03 |  |  |  |
| `rms_energy` | 0.36 | 0.02 |  |  | 0.08 |
| `spectral_centroid` | 8.27 | 0.19 | 8.71 | `feature.spectral_centroid` | 5.88 |
| `spectral_bandwidth` | 10.41 | 1.76 | 12.44 | `feature.spectral_bandwidth` | 6.86 |
| `spectral_rolloff` | 9.57 | 0.36 | 8.91 | `feature.spectral_rolloff` | 7.88 |
| `spectral_contrast` | 10.94 | 0.81 | 8.67 | `feature.spectral_contrast` | 18.71 |
| `spectral_entropy` | 10.24 | 0.38 |  |  | 7.35 |
| `spectral_flatness` | 9.39 | 0.44 | 6.74 | `feature.spectral_flatness` | 6.54 |
| `spectral_flux` | 11.23 | 0.34 |  |  | 5.68 |
| `spectral_skewness` | 12.36 | 0.54 |  |  | 10.26 |
| `spectral_slope` | 9.16 | 0.17 |  |  | 7.38 |
| `spectral_kurtosis` | 10.84 | 0.61 |  |  | 16.96 |
| `spectral_crest` | 8.90 | 0.78 |  |  | 6.85 |
| `spectral_decrease` | 9.32 | 0.40 |  |  | 5.91 |
| `melspectrogram` | 9.75 | 0.43 | 8.26 | `feature.melspectrogram` | 5.85 |
| `mfcc` | 9.98 | 0.62 | 8.48 | `feature.mfcc` | 6.14 |
| `mfcc_delta` | 10.59 | 0.77 | 10.10 | `feature.delta(mfcc)` | 6.28 |
| `mfcc_delta_delta` | 10.97 | 0.72 | 9.16 | `feature.delta(mfcc, order=2)` | 7.18 |
| `chroma_stft` | 9.68 | 0.86 | 7.21 | `feature.chroma_stft(tuning=0)` | 5.53 |
| `chroma_cqt` | 9.72 | 0.55 | 73.64 | `feature.chroma_cqt *` |  |
| `tonnetz` | 9.60 | 1.02 | 7.45 | `feature.tonnetz(chroma=chroma_stft)` |  |
| `onset_strength` | 9.98 | 0.50 | 8.25 | `onset.onset_strength` |  |
| `cqt` | 9.46 | 0.34 | 21.82 | `cqt *` |  |
| `pitch` | 19.81 | 0.54 | 22.98 | `yin *` |  |
| `thd` | 3.71 | 0.18 |  |  |  |
| `hnr` | 18.05 | 0.44 |  |  |  |
| `lpc` | 11.93 | 0.48 |  |  |  |
| `formants` | 46.31 | 2.99 |  |  |  |
| `gfcc` | 9.69 | 0.48 |  |  | 6.36 |
| total | 621.81 |  |  |  | 136.18 (one module) |

`extract_all_features` on the same config, sharing one STFT per framing: 565.46 ms

## Reading the table

Rows marked `*` are not the same computation. AFX's `cqt` and `chroma_cqt`
weight STFT bins with log-spaced triangles, so they cost one STFT; librosa
computes a true constant-Q transform with octave-wise filtering and
resampling, which is 2 to 8 times slower here and has a time resolution
that varies with frequency. `pitch` and `librosa.yin` implement the same
estimator but differ in how unvoiced frames are reported (see
[librosa equivalence](librosa.md)).

For the rows that are numerically equal to librosa, the two libraries are
within about 40% of each other (`zcr` aside), because both are dominated
by one real FFT over about 430 frames of 2048 samples. AFX also memoizes
the mel and gammatone filterbanks across calls; librosa rebuilds its mel
filterbank on every call.

`sample_entropy` is over 40% of the total. Its cost grows with the
square of the frame length in the worst case (Richman and Moorman, 2000), and the
bundled config computes it on frames of 1024 samples. Drop it, or use
shorter frames, when throughput matters.

`extract_all_features` reuses one STFT for features with the same
framing, centering and padding mode. With the bundled config that saves
about 55 ms of the 620 ms sum; the rest of each spectral feature's time is
the magnitude, power and per-feature reductions, which are not shared.

On this two-core machine the torch path runs most spectral features at
the same speed per clip as the NumPy path or up to 2 times faster
(`spectral_flux` 5.7 ms against 11.2 ms) by batching the FFT and matrix
products. `spectral_contrast` (a sort per octave band), `spectral_kurtosis`
and `zcr` are slower in torch. Expect the gap to change with core count and on a GPU; rerun the
command above on the target machine.

## Variance

On a shared VM, single runs move by 10 to 30%: runs of the command above
on the same machine gave 622 to 678 ms for the total. The IQR column shows the spread within one run.
Compare libraries within one run of the script, not across runs or
machines.
