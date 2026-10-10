---
title: 'AFX: configurable audio feature extraction for machine learning with NumPy, scikit-learn and PyTorch front ends'
tags:
  - Python
  - audio
  - feature extraction
  - signal processing
  - machine learning
  - speech
  - bioacoustics
authors:
  - name: Danilo Pena
    orcid: 0000-0003-2767-5764
    affiliation: 1
affiliations:
  - name: Affiliation to be filled in, Country
    index: 1
date: 10 October 2026
bibliography: paper.bib
---

<!-- Maintainer: replace the affiliation above before submission. -->

# Summary

AFX is a Python library that turns audio clips into fixed-length feature
vectors for statistical learning. It implements 47 features on NumPy and
SciPy [@harris2020numpy; @virtanen2020scipy]: 45 descriptors in seven
groups (time domain, autocorrelation, spectral, spectrogram, cepstral,
harmonic and linear prediction) and the eGeMAPS v02 parameter set
[@eyben2016gemaps]. A JSON configuration selects the features and their
framing parameters, the configuration is checked before any audio is
read, and per-frame values are reduced to per-clip statistics. The same
configuration drives a scikit-learn transformer [@pedregosa2011sklearn],
a parallel batch extractor with a command-line interface, and, for 20 of
the features, a differentiable PyTorch module [@paszke2019pytorch] whose
output matches the NumPy path. Batch outputs record the configuration
hash, library versions and input file hashes they were made from. The
core install depends only on NumPy, SciPy and soundfile, and the code is
MIT licensed.

# Statement of need

Hand-crafted acoustic features remain the input of choice where data
sets are small, models must be inspected, or results have to be compared
with earlier studies: voice and speech pathology, animal vocalisation
studies, machine condition monitoring, and baselines for deep models.
These studies need three things that general audio libraries leave to
the user. The first is a declarative description of the features and
parameters, so that an extraction can be repeated and reported exactly.
The second is a fixed-length vector per clip with named columns and a
defined treatment of frames where a feature is undefined (the
harmonics-to-noise ratio of silence, the formants of noise). The third
is the same numbers whether features are computed for a scikit-learn
pipeline or inside a neural network, so that a classical baseline and a
learned front end can share a definition.

AFX addresses these with a single configuration format, versioned
presets, an aggregation layer that skips undefined (NaN) frames and can
report how many frames each statistic rests on, column names that are
stable across the NumPy, scikit-learn and PyTorch paths, and tests that
pin those paths to each other, to librosa and to openSMILE. Its target
users are researchers who build classifiers or regressors on audio and
want a permissively licensed, dependency-light extractor whose
definitions are documented to the formula.

# State of the field

librosa [@mcfee2015librosa] (ISC licence) is the reference Python
library for music and audio analysis. It computes frame-wise
representations and descriptors but leaves feature selection,
aggregation to clip level and batch processing to the caller. AFX
reproduces 14 librosa features to float32 precision, uses the same
centered framing by default, and builds the missing layers on top.
torchaudio [@yang2022torchaudio] (BSD-2-Clause) provides differentiable
spectrograms, MFCC and a few spectral descriptors for PyTorch; it
entered a maintenance phase with version 2.9 and removed the APIs
deprecated in 2.8 [@torchaudio2025maintenance]. nnAudio
[@cheuk2020nnaudio] (MIT) computes spectrograms, including a constant-Q
transform, on the GPU with optionally trainable kernels, and diffsptk,
the differentiable PyTorch counterpart of SPTK [@yoshimura2023sptk4]
(Apache-2.0), covers speech analysis and synthesis operators. These
three are representation layers rather than descriptor sets with
clip-level aggregation.

Essentia [@bogdanov2013essentia] is a C++ library with Python bindings
and a large set of music descriptors, licensed under the AGPL-3.0, which
restricts use in closed or networked software. openSMILE
[@eyben2010opensmile] defines the standard paralinguistic feature sets,
including eGeMAPS [@eyben2016gemaps]; it is free for research,
personal and educational use, and commercial use needs a licence from
audEERING. pyAudioAnalysis [@giannakopoulos2015pyaudioanalysis]
(Apache-2.0) combines short-term features with classification and
segmentation tools. Surfboard [@lenain2020surfboard] targeted the same
clip-level use case for speech and health applications, but is
GPL-3.0 licensed and its repository was archived in January 2025. TSFEL
[@barandas2020tsfel] (BSD-3-Clause) extracts more than 65 features from
generic time series, without audio-specific framing conventions.

AFX was written as a separate package rather than as contributions to
these projects because its core is a configuration, validation and
aggregation contract that spans three front ends, which is outside the
scope of librosa and of the representation libraries, and because a
permissive licence was a requirement for the intended users.

# Software design and functionality

Every feature is a function `extract_<name>(signal, sr, **params)` that
returns a dictionary of float64 arrays, registered under its name in a
feature map. A registry records each feature's group, units, output
shape, parameters and defaults, the condition under which it is
undefined, and its PyTorch counterpart; `feature_info` prints this
record and the documentation's feature reference is generated from it.
`register_feature` adds user extractors, which then work in
configurations and in every front end. Docstrings give the formula of
each feature and cite its source, for example MFCC [@davis1980mfcc], YIN
pitch [@decheveigne2002yin], the harmonics-to-noise ratio
[@boersma1993hnr], sample entropy [@richman2000sampen], Hjorth
parameters [@hjorth1970], spectral contrast [@jiang2002contrast], tonal
centroids [@harte2006tonnetz], gammatone cepstra
[@shao2008gfcc; @glasberg1990erb], an STFT-based approximation of the
constant-Q transform [@brown1991cqt] and SuperFlux onset strength
[@bock2013superflux].

Every framed feature defaults to centered frames, as in librosa, so
features with the same frame size and hop share frame count and frame
times. A frame where a feature has no defined value is NaN (pitch on an
unvoiced frame, spectral entropy on silence); aggregation skips such
frames, and a statistic is NaN only when every frame is undefined.
Setting `"coverage": true` adds a `<feature>_valid_frac` column with the
fraction of frames each statistic is based on, so that clip-level values
computed from few valid frames can be identified or filtered downstream.

`validate_config` rejects unknown feature names, unknown parameters and
non-positive frame sizes before extraction starts, so a misspelled
parameter fails instead of falling back to a default. Five presets
(speech, bioacoustics, music, machinery and eGeMAPS) provide starting
points. They are versioned files that spell out every parameter; an id
such as `speech@1` names one version, and a test pins the configuration
hash of each released version, so a later change to a default ships as a
new version. Aggregation reduces each per-frame feature to its mean,
standard deviation, higher moments, sliding-window statistics or a
stacked matrix, and `flatten_features` returns column names and one
vector. Extractors with identical STFT parameters share one memoised
transform.

`AFXFeatureExtractor` is a scikit-learn transformer that accepts file
paths, arrays or a preset id, extracts in parallel and implements
`get_feature_names_out`. `afx_audio.torch` reimplements 20 features (mel
spectrogram, MFCC and deltas, gammatone cepstra, chroma, twelve spectral
descriptors, zero-crossing rate and RMS) as batched, device-agnostic
PyTorch operations that are differentiable with respect to the
waveform, except through the counting in the zero-crossing rate and the
bin selection in the spectral roll-off. `AFXFeatureModule` wraps them as
an `nn.Module` configured like the NumPy path. The `afx` command and
`extract_features_from_folder` process a directory tree in worker
processes that are never forked, and write NumPy, JSON, CSV or Parquet
output; files that fail are listed in a separate error file, and
parallel output is identical to serial output.

AFX includes a reimplementation of the eGeMAPS v02 parameter set
[@eyben2016gemaps]: 25 low-level descriptors every 10 ms and 88
functionals named and ordered as in openSMILE's output, written from the
published definition and openSMILE's configuration parameters without
openSMILE code, so that the set is available under the MIT licence. F0
uses subharmonic summation [@hermes1988shs]. openSMILE 2.6.0 runs only
in tests and a validation script that report agreement per functional on
20 synthetic voice clips and 13 three-second LibriSpeech segments.
Loudness, spectral flux, MFCC, equivalent sound level and the F0 level
statistics agree within 3 % on both sets (Pearson r of 0.99 or more
across clips); jitter, shimmer and F0 slopes differ by 10 to 50 %, since
they depend on undocumented voicing and period decisions. Formants
differ by design: AFX discards LPC poles with a bandwidth of 1 kHz or
more, and on the synthetic clips its mean F1, F2 and F3 are within 18,
38 and 86 Hz of the synthesis values, against 245, 803 and 483 Hz for
openSMILE.

Batch outputs carry a provenance record: AFX, NumPy, SciPy, soundfile
and Python versions, the validated configuration and its hash (the first
12 hex characters of the SHA-256 of its canonical JSON), the preset id
and whether an override changed it, resampling details and the SHA-256
of every input file. Parquet files keep it in the schema metadata, other
formats in a sidecar file, and the scikit-learn transformer exposes the
same record after fitting. During extraction a streaming accumulator
computes per-column count, mean, variance, minimum, maximum and NaN
count [@welford1962], merging worker shards with the pairwise update of
@chan1983variance; `scaler_from_stats` turns the result into a fitted
scikit-learn `StandardScaler` without loading the feature matrix.

# Quality control

The test suite (over 1100 tests, run with pytest) checks each extractor
against analytic cases (pure tones, white noise, gain and power-law
scaling, time reversal), runs every feature on silence, constant,
one-sample and very short input, checks that non-finite, empty, integer
and multichannel input is rejected, and checks that every output is
float64 and every warning is an `AFXWarning`.
`tests/test_librosa_parity.py` compares 14 features with librosa at
four framings and requires agreement to float32 precision. The PyTorch
functions are compared with the NumPy path in float64 to a relative
tolerance of $10^{-9}$ at two framings, batched and unbatched, with the
tolerances read from the same table that `feature_info` reports, and
gradients are checked to propagate to the waveform. The eGeMAPS tests
compare with openSMILE when it is installed. Continuous integration runs
the suite on Linux with Python 3.10, 3.12 and 3.14 and on Windows and
macOS with Python 3.13, repeats it with the oldest supported dependency
versions, executes the tutorial notebooks, lints with ruff, type-checks
with mypy, rebuilds the documentation and fails when the generated
feature reference is stale, and installs the built wheel in a clean
environment. A benchmark script times each feature against its librosa
counterpart and the PyTorch path. On a two-core 2.1 GHz virtual machine,
39 features of the bundled configuration take 0.57 s for 10 s of
22.05 kHz audio, spectral features run within about 40 % of librosa's
time, and the PyTorch module computes its 20 features at 0.14 s per clip
in a batch of 16.

# Research impact statement

<!-- Maintainer: describe realized or near-term use (projects, papers, downloads, users) with specific evidence, as JOSS requires. -->

# AI usage disclosure

<!-- Maintainer: JOSS requires a statement of how generative AI tools were used in the software, documentation or paper, or that none were. -->

# Acknowledgements

<!-- Maintainer: funding sources and acknowledgements. -->

# References
