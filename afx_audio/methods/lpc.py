"""
Linear prediction by the autocorrelation method.
"""
import numpy as np

from afx_audio.methods.autocorrelation import autocorrelation


def lpc(frames: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """
    LPC polynomials of each row of ``frames`` by Levinson-Durbin recursion.

    Returns (a, err): ``a`` has shape (n_frames, order + 1) with a[:, 0] = 1,
    the coefficients of A(z) = 1 + a_1 z^-1 + ... + a_p z^-p that minimize
    the prediction error of x[n] + sum_k a_k x[n-k] (the convention of
    librosa.lpc, which uses Burg's method instead); ``err`` is the final
    prediction error power. Frames are used as given, so window them first.
    A silent frame gives A(z) = 1.
    """
    frames = np.atleast_2d(np.asarray(frames, dtype=np.float64))
    n = frames.shape[1]
    if order < 1 or order >= n:
        raise ValueError(f'order must be between 1 and frame length - 1 ({n - 1}), got {order}')
    return levinson(autocorrelation(frames, order + 1), order)


def levinson(r: np.ndarray, order: int) -> tuple[np.ndarray, np.ndarray]:
    """Levinson-Durbin recursion on autocorrelations r[:, 0..order], returning (a, err) as lpc does."""
    r = np.atleast_2d(np.asarray(r, dtype=np.float64))
    n_frames = len(r)
    a = np.zeros((n_frames, order + 1))
    a[:, 0] = 1.0
    err = r[:, 0].copy()
    for i in range(1, order + 1):
        acc = r[:, i] + np.einsum('fj,fj->f', a[:, 1:i], r[:, i - 1:0:-1])
        k = np.divide(-acc, err, out=np.zeros_like(acc), where=err > 0)
        a[:, 1:i] = a[:, 1:i] + k[:, None] * a[:, i - 1:0:-1]
        a[:, i] = k
        err = err * (1.0 - k * k)
    return a, err


def lpc_roots(a: np.ndarray) -> np.ndarray:
    """Roots of each A(z) in ``a`` (n_frames, order + 1), shape (n_frames, order), from batched companion matrices."""
    a = np.atleast_2d(a)
    n_frames, p1 = a.shape
    order = p1 - 1
    companion = np.zeros((n_frames, order, order))
    companion[:, 0, :] = -a[:, 1:]
    companion[:, np.arange(1, order), np.arange(order - 1)] = 1.0
    return np.linalg.eigvals(companion)
