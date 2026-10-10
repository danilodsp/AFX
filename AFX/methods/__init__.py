"""
Modular signal processing methods for audio feature extraction.
"""
from .autocorrelation import autocorrelation
from .chroma import (
    chroma_filterbank,
    chroma_from_power,
    chroma_from_stft,
    extract_chroma_from_cqt,
    frequency_to_chroma_bin,
)
from .cqt import cqt_approximation, cqt_filterbank
from .dct import extract_mfcc_coefficients
from .delta import compute_delta, delta_kernel
from .gammatone import apply_gammatone_filterbank, erb_space, gammatone_filterbank
from .lpc import lpc, lpc_roots
from .mel import (
    apply_mel_filterbank,
    hz_to_mel,
    log_mel_spectrogram,
    mel_filterbank,
    mel_frequencies,
    mel_to_hz,
)
from .stft import clear_stft_cache, magnitude_spectrogram, power_spectrogram, stft

__all__ = [
    'stft', 'magnitude_spectrogram', 'power_spectrogram', 'clear_stft_cache',
    'autocorrelation',
    'hz_to_mel', 'mel_to_hz', 'mel_frequencies', 'mel_filterbank',
    'apply_mel_filterbank', 'log_mel_spectrogram',
    'extract_mfcc_coefficients',
    'delta_kernel', 'compute_delta',
    'cqt_filterbank', 'cqt_approximation',
    'chroma_filterbank', 'chroma_from_power', 'chroma_from_stft', 'extract_chroma_from_cqt', 'frequency_to_chroma_bin',
    'lpc', 'lpc_roots',
    'erb_space', 'gammatone_filterbank', 'apply_gammatone_filterbank',
]
