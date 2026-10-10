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
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Affiliation to be filled in, Country
    index: 1
date: 10 October 2026
bibliography: paper.bib
---

<!-- Maintainer: replace the ORCID and the affiliation above before submission. -->

# Summary

AFX is a Python library that turns audio clips into fixed-length feature
vectors for statistical learning. It implements 45 features in seven
groups (time domain, autocorrelation, spectral, spectrogram, cepstral,
harmonic and linear prediction) on NumPy and SciPy
[@harris2020numpy; @virtanen2020scipy]. A JSON configuration selects the
features and their framing parameters, the configuration is checked
before any audio is read, and per-frame values are reduced to per-clip
statistics. The same configuration drives a scikit-learn transformer
[@pedregosa2011sklearn], a parallel batch extractor with a command-line
interface, and, for 20 of the features, a differentiable PyTorch module
[@paszke2019pytorch] whose output matches the NumPy path. The core
install depends only on NumPy, SciPy and soundfile, and the code is MIT
licensed.

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

AFX addresses these with a single configuration format, an aggregation
layer that skips undefined (NaN) frames, column names that are stable
across the NumPy, scikit-learn and PyTorch paths, and tests that pin
those paths to each other and to librosa. Its target users are
researchers who build classifiers or regressors on audio and want a
permissively licensed, dependency-light extractor whose definitions are
documented to the formula.

# State of the field

librosa [@mcfee2015librosa] (ISC licence) is the reference Python
library for music and audio analysis. It computes frame-wise
representations and descriptors but leaves feature selection,
aggregation to clip level and batch processing to the caller. AFX
reproduces 14 librosa features to float32 precision and builds the
missing layers on top. torchaudio [@yang2022torchaudio] (BSD-2-Clause)
provides differentiable spectrograms, MFCC and a few spectral
descriptors for PyTorch; it entered a maintenance phase with version 2.9
and removed the APIs deprecated in 2.8 [@torchaudio2025maintenance].
nnAudio [@cheuk2020nnaudio] (MIT) computes spectrograms, including a
constant-Q transform, on the GPU with optionally trainable kernels, and
diffsptk, the differentiable PyTorch counterpart of SPTK
[@yoshimura2023sptk4] (Apache-2.0), covers speech analysis and synthesis
operators. These three are representation layers rather than descriptor
sets with clip-level aggregation.

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
returns a dictionary of arrays, registered under its name in a feature
map. A registry records each feature's group, units, output shape,
parameters and defaults, which `feature_info` prints and from which the
documentation's feature reference is generated. `register_feature` adds
user extractors, which then work in configurations and in every front
end. Docstrings give the formula of each feature and cite its source,
for example MFCC [@davis1980mfcc], YIN pitch [@decheveigne2002yin], the
harmonics-to-noise ratio [@boersma1993hnr], sample entropy
[@richman2000sampen], Hjorth parameters [@hjorth1970], spectral contrast
[@jiang2002contrast], tonal centroids [@harte2006tonnetz], gammatone
cepstra [@shao2008gfcc; @glasberg1990erb], an STFT-based approximation
of the constant-Q transform [@brown1991cqt] and SuperFlux onset
strength [@bock2013superflux].

`validate_config` rejects unknown feature names, unknown parameters and
non-positive frame sizes before extraction starts, so a misspelled
parameter fails instead of falling back to a default. Four presets
(speech, bioacoustics, music, machinery) provide starting points.
Aggregation reduces each per-frame feature to its mean, standard
deviation, higher moments, sliding-window statistics or a stacked
matrix, ignoring NaN frames, and `flatten_features` returns column names
and one vector. Extractors with identical STFT parameters share one
memoised transform.

`AFXFeatureExtractor` is a scikit-learn transformer that accepts file
paths or arrays, extracts in parallel and implements
`get_feature_names_out`. `afx_audio.torch` reimplements 20 features (mel
spectrogram, MFCC and deltas, gammatone cepstra, chroma, twelve spectral
descriptors, zero-crossing rate and RMS) as batched, device-agnostic
PyTorch operations that are differentiable with respect to the
waveform, except through the counting in the zero-crossing rate and the
bin selection in the spectral roll-off. `AFXFeatureModule` wraps them as
an `nn.Module` configured like the NumPy path. The `afx` command and
`extract_features_from_folder` process a directory tree in parallel and
write NumPy, JSON, CSV or Parquet output; files that fail are listed in
a separate error file, and parallel output is identical to serial
output.

<!-- confirm when merged -->
AFX includes a reimplementation of the eGeMAPS v02 parameter set
[@eyben2016gemaps], 25 low-level descriptors every 10 ms and 88
functionals named and ordered as in openSMILE's output, written from the
published definition and openSMILE's configuration parameters without
openSMILE code, so that the set is available under the MIT licence.
openSMILE is used only in tests and a validation script that measure
agreement per functional.

<!-- confirm when merged -->
Batch outputs carry a provenance record (library and dependency
versions, the validated configuration and its hash, resampling details
and per-file SHA-256 digests), stored in Parquet metadata or a sidecar
file, and the scikit-learn transformer exposes the same record after
fitting. A streaming accumulator computes per-column dataset statistics
during extraction.

<!-- confirm when merged -->
An opt-in coverage column per feature reports the fraction of frames
with a finite value, so that clip-level statistics computed from few
valid frames can be identified or filtered downstream.

# Quality control

The test suite (over 700 tests, run with pytest) checks each extractor
against analytic cases (pure tones, white noise, gain and power-law
scaling, time reversal), runs every feature on silence, constant,
one-sample and very short input without warnings, and checks that
non-finite, empty, integer and multichannel input is rejected. `tests/test_librosa_parity.py` compares 14
features with librosa at four framings and requires agreement to float32
precision. The PyTorch functions are compared with the NumPy path in
float64 to a relative tolerance of $10^{-6}$ or tighter, batched and
unbatched, and gradients are checked to propagate to the waveform.
Continuous integration runs the suite on Linux with Python 3.10, 3.12
and 3.14 and on Windows and macOS with Python 3.13, repeats it with the
oldest supported dependency versions, lints with ruff, type-checks with
mypy, rebuilds the documentation and fails when the generated feature
reference is stale, and installs the built wheel in a clean environment.
A benchmark script times each feature against its librosa counterpart
and the PyTorch path on the same machine.

# Research impact statement

<!-- Maintainer: describe realized or near-term use (projects, papers, downloads, users) with specific evidence, as JOSS requires. -->

# AI usage disclosure

<!-- Maintainer: JOSS requires a statement of how generative AI tools were used in the software, documentation or paper, or that none were. -->

# Acknowledgements

<!-- Maintainer: funding sources and acknowledgements. -->

# References
