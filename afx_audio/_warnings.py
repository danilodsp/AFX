"""Warning categories, so AFX warnings can be filtered apart from those of NumPy and SciPy."""
import warnings


class AFXWarning(UserWarning):
    """Base category of every warning AFX emits."""


class AFXDeprecationWarning(AFXWarning, DeprecationWarning):
    """An AFX name or parameter that will be removed; shown by default only for calls from ``__main__``."""


def warn_deprecated(message: str, *, stacklevel: int = 3) -> None:
    """Warn with AFXDeprecationWarning; the default stacklevel points at the caller of the deprecated function."""
    warnings.warn(message, AFXDeprecationWarning, stacklevel=stacklevel)
