# AFX (Audio Feature Extraction)

This project can be used for audio research in fields such as machine learning, sound classification, speech analysis, and music information retrieval.

---

## Features List

- **Standard Audio Features**
  - Mel-Spectrogram
  - MFCC (Mel-Frequency Cepstral Coefficients)
  - Delta MFCC
  - Delta-Delta MFCC
  - Zero Crossing Rate (ZCR)
  - Variance
  - Entropy (Sample Entropy)
  - Crest Factor
  - Spectral Centroid
  - Spectral Bandwidth
  - Spectral Rolloff
  - Chroma Features (STFT-based)
  - Chroma Features (CQT-based)
  - Constant-Q Transform (CQT)
  - Pitch (Fundamental Frequency)
  - Total Harmonic Distortion (THD)
  - Harmonic to Noise Ratio (HNR)

- **Additional Implemented Features**
  - Spectral Contrast
  - Spectral Flatness
  - Spectral Entropy
  - Spectral Flux
  - Kurtosis
  - Short-Time Energy
  - Energy Ratio
  - Autocorrelation-based features (variance, SNR, zero crossings, peak width, decay rate, coeffs deviation, max peak envelope)
  - Mobility
  - Complexity
  - Sample Entropy (alternative implementation)

- **Support for ML/DL Pipelines**
  - Frame-level and file-level feature aggregation, with options for flattening, stacking, and normalization (z-score, min-max)
  - Compatible with numpy arrays and torch tensors
  - Custom windowing and hop-length settings

- **I/O & Formats**
  - WAV, MP3, FLAC, OGG (via `soundfile`)
  - Batch processing via folder-based loaders

---

## Project Structure

```
audio-features/
│
├── AFX/                      # Core package
│   ├── extractors/           # Each group features extractor is a modular class
│   ├── io/                   # Audio loading and format handling
│   ├── methods/              # Signal processing algorithms
│   ├── utils/                # Helpers (e.g. windowing, normalization)
│   ├── pipelines/            # Predefined feature extraction pipelines
│   └── __init__.py
│
├── notebooks/                # Jupyter notebooks for examples
├── examples/                 # Script examples
├── tests/                    # Unit tests
├── requirements.txt
├── README.md
└── setup.py
```

---

## Getting Started

```bash
git clone https://github.com/danilodsp/AFX.git
cd AFX

python -m venv venv
source venv\Scripts\activate

pip install -r requirements.txt
```

---


## Example Usage

```python
from AFX import extract_all_features
from AFX.utils.aggregator import aggregate_features

# Extract frame-level features
features = extract_all_features('path/to/audio.wav', sr=22050)

# Aggregate, flatten, and z-score normalize features in one step
agg_features = aggregate_features(features, method='mean', flatten=True, normalize='zscore')

# Stack all features into a single vector (option 1: via features_to_vector)
from AFX.utils import features_to_vector
feature_vector = features_to_vector(agg_features, flatten=False)
print(feature_vector.shape)

# Stack all features directly using the aggregation pipeline (option 2)
stacked_vector = aggregate_features(features, method='stack')
print(stacked_vector.shape)
```

Check the [Notebooks](notebooks/) folder for more examples.

## Acknowledgment

- Audio samples used in this project for unit testing purposes are derived from the LibriSpeech ([Panayotov et al., 2015](https://www.danielpovey.com/files/2015_icassp_librispeech.pdf)) which is based on public domain ([download link](https://www.openslr.org/12)).
