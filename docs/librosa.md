# librosa equivalence

`tests/test_librosa_parity.py` compares these features with librosa at
four framings (22.05 kHz with n_fft 2048 / hop 512, 16 kHz with 400 / 160
and 512 / 128, centered and not). Agreement is to float32 precision or
better.

| AFX feature | librosa call |
|---|---|
| `spectral_centroid` | `feature.spectral_centroid` |
| `spectral_bandwidth` | `feature.spectral_bandwidth` |
| `spectral_rolloff` | `feature.spectral_rolloff` |
| `spectral_flatness` | `feature.spectral_flatness` |
| `spectral_contrast` | `feature.spectral_contrast` |
| `melspectrogram` | `feature.melspectrogram(pad_mode='constant')` |
| `mfcc` | `feature.mfcc(pad_mode='constant')` |
| `mfcc_delta`, `mfcc_delta_delta` | `feature.delta(mfcc, order=1 / 2)` |
| `chroma_stft` | `feature.chroma_stft(tuning=0.0)` |
| `tonnetz` | `feature.tonnetz(chroma=feature.chroma_stft(tuning=0.0))` |
| `onset_strength` | `onset.onset_strength` |
| `zcr` | `feature.zero_crossing_rate` |
| `rms_energy` (framed) | `feature.rms` |

Use the same `n_fft`/`frame_size`, `hop_length` and `center` on both
sides. librosa pads centered frames with zeros (`pad_mode='constant'`)
since 0.10, older versions reflected; `zcr` repeats the edge samples on
both sides, as `librosa.feature.zero_crossing_rate` does.

## Differences

`pitch` agrees with `librosa.yin(center=False)` to within 2% on voiced
frames (librosa 0.10 and 1.0 differ from each other by about 1%). AFX
returns 0 for frames without a clear periodicity dip, where librosa
always reports its best lag, which can be anywhere in the range (about
fmax on silence).

`chroma_stft` takes the tuning as a parameter (default 0, A440). librosa
estimates it from the signal when `tuning` is not given, which shifts the
filterbank on detuned recordings.

`cqt` and `chroma_cqt` are computed from the STFT through a triangular
log-frequency filterbank. They have the time resolution of the STFT at
every frequency and do not match `librosa.cqt`, which uses true
constant-Q filters.

`zcr` and the other time-domain features default to `center=False`, while
librosa's `zero_crossing_rate` and `rms` default to `center=True`; pass the
same value on both sides.

`spectral_flux` is the Euclidean distance between consecutive magnitude
spectra, not librosa's onset strength.
