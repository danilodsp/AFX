"""
Unit test for 2D audio features (mel spectrogram, MFCC) using a real .wav file and comparison with Librosa.
"""
import numpy as np
import pytest
import soundfile as sf
from AFX.methods.stft import stft, power_spectrogram
from AFX.methods.mel import mel_filterbank, apply_mel_filterbank, log_mel_spectrogram
from AFX.methods.dct import extract_mfcc_coefficients

import librosa
import os

TEST_WAV = "tests/data/116-288045-0000.wav"

@pytest.mark.skipif(not os.path.exists(TEST_WAV), reason="test audio file not available")
@pytest.mark.parametrize("sr,n_mels,n_mfcc,frame_size,hop_length", [
    (16000, 40, 13, 1024, 512),
    (22050, 40, 13, 2048, 512),
])
def test_mel_and_mfcc_against_librosa(sr, n_mels, n_mfcc, frame_size, hop_length):
    """
    Compare AFX mel spectrogram and MFCC with Librosa on a real audio file.
    """
    # Load audio
    y, file_sr = sf.read(TEST_WAV)
    if y.ndim > 1:
        y = y.mean(axis=1)  # Convert to mono if needed
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)

    # --- Librosa reference ---
    mel_librosa = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=frame_size, hop_length=hop_length, n_mels=n_mels, power=2.0)
    log_mel_librosa = librosa.power_to_db(mel_librosa, ref=np.max)
    mfcc_librosa = librosa.feature.mfcc(S=log_mel_librosa, n_mfcc=n_mfcc)

    # --- AFX pipeline ---
    stft_result = stft(y, frame_size=frame_size, hop_length=hop_length)
    power_spec = power_spectrogram(stft_result)
    filterbank = mel_filterbank(n_mels, frame_size, sr)
    mel_spec = apply_mel_filterbank(power_spec, filterbank)
    log_mel = log_mel_spectrogram(mel_spec)
    # Normalize log-mel to dB scale similar to librosa
    log_mel_db = log_mel - np.max(log_mel)
    mfcc = extract_mfcc_coefficients(log_mel, n_mfcc)

    # --- Compare shapes ---
    assert mel_spec.shape == mel_librosa.shape
    assert mfcc.shape == mfcc_librosa.shape

    # --- Compare values (allowing for small numerical differences) ---
    np.testing.assert_allclose(log_mel_db, log_mel_librosa, rtol=1e-2, atol=1.0)

    # Exclude the first MFCC coefficient from strict comparison
    np.testing.assert_allclose(
        mfcc[1:], mfcc_librosa[1:], rtol=1e-2, atol=1.0,
        err_msg="MFCC coefficients 1..n do not match Librosa within tolerance"
    )

    # Only check the first coefficient is finite (do not compare value)
    assert np.isfinite(mfcc[0]).all() and np.isfinite(mfcc_librosa[0]).all(), "First MFCC coefficient contains non-finite values"

    # --- Statistical comparison ---
    def describe(arr, name):
        return f"{name}: mean={np.mean(arr):.2f}, std={np.std(arr):.2f}, min={np.min(arr):.2f}, max={np.max(arr):.2f}"

    print(describe(log_mel_db, "AFX log-mel"))
    print(describe(log_mel_librosa, "Librosa log-mel"))
    print(describe(mfcc[1:], "AFX MFCC[1:]") )
    print(describe(mfcc_librosa[1:], "Librosa MFCC[1:]") )

    # --- Correlation analysis ---
    def corr2d(a, b):
        a_flat = a.flatten()
        b_flat = b.flatten()
        return np.corrcoef(a_flat, b_flat)[0, 1]

    logmel_corr = corr2d(log_mel_db, log_mel_librosa)
    mfcc_corr = corr2d(mfcc[1:], mfcc_librosa[1:])
    print(f"Log-mel correlation: {logmel_corr:.4f}")
    print(f"MFCC[1:] correlation: {mfcc_corr:.4f}")

    assert logmel_corr > 0.98, f"Log-mel correlation too low: {logmel_corr:.4f}"
    assert mfcc_corr > 0.95, f"MFCC[1:] correlation too low: {mfcc_corr:.4f}"

    # Check for finite values
    assert np.isfinite(mel_spec).all()
    assert np.isfinite(mfcc).all()
