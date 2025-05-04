# AFX (Audio Features Extraction)

This is a Python module for extracting features from audio signals. It can be used for audio research in fields such as machine learning, sound classification, speech analysis, and music information retrieval.

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
  - Frame-level and file-level feature aggregation
  - Compatible with numpy arrays and torch tensors
  - Custom windowing and hop-length settings

- **I/O & Formats**
  - WAV, MP3, FLAC, OGG support via `librosa` and `soundfile`
  - Batch processing via folder-based loaders

---

## Project Structure

```
audio-features/
│
├── AFX/           # Core package
│   ├── extractors/           # Each group features extractor is a modular class
│   ├── io/                   # Audio loading and format handling
│   ├── utils/                # Helpers (e.g. windowing, normalization)
│   ├── pipelines/            # Predefined feature extraction pipelines
│   └── __init__.py
│
├── notebooks/                # Jupyter notebooks for examples/tutorials
├── examples/                 # Script examples
├── tests/                    # Unit tests
├── requirements.txt
├── README.md
└── setup.py
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/danilodsp/AFX.git
cd audio-features
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Example Usage

```python
from AFX import extract_all_features

features = extract_all_features('path/to/audio.wav', sr=22050)
print(features['mfcc'].shape)  # e.g., (13, T)
```

For more, check the [Notebooks](notebooks/) folder for more examples.
