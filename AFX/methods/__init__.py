"""
Modular signal processing methods for audio feature extraction.
"""
from .stft import stft, magnitude_spectrogram, power_spectrogram, clear_stft_cache
from .mel import (
    hz_to_mel, mel_to_hz, mel_frequencies, mel_filterbank,
    apply_mel_filterbank, log_mel_spectrogram,
)
from .dct import extract_mfcc_coefficients
from .delta import delta_kernel, compute_delta
from .cqt import cqt_filterbank, cqt_approximation, pseudo_cqt
from .chroma import chroma_from_stft, extract_chroma_from_cqt, frequency_to_chroma_bin
from .gammatone import erb_space, gammatone_filterbank, apply_gammatone_filterbank

__all__ = [
    'stft', 'magnitude_spectrogram', 'power_spectrogram', 'clear_stft_cache',
    'hz_to_mel', 'mel_to_hz', 'mel_frequencies', 'mel_filterbank',
    'apply_mel_filterbank', 'log_mel_spectrogram',
    'extract_mfcc_coefficients',
    'delta_kernel', 'compute_delta',
    'cqt_filterbank', 'cqt_approximation', 'pseudo_cqt',
    'chroma_from_stft', 'extract_chroma_from_cqt', 'frequency_to_chroma_bin',
    'erb_space', 'gammatone_filterbank', 'apply_gammatone_filterbank',
]
