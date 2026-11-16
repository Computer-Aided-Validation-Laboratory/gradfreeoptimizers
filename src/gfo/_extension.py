"""Loads the compiled C++ extension (gfo._pso) with a helpful error if missing.

``import gfo`` works without the extension (the Python example problems and
post-processing do not need it), only running an optimizer does.
"""

from types import ModuleType

BUILD_HINT = ("The C++ extension gfo._pso is not built (or failed to load). "
              "Build it with: python scripts/build.py")

try:
    from gfo import _pso as _extension
    IMPORT_ERROR: ImportError | None = None
except ImportError as exc:  # not built yet, or built for another Python
    _extension = None
    IMPORT_ERROR = exc

EXTENSION_AVAILABLE = _extension is not None


def get_extension() -> ModuleType:
    """Return the compiled ``gfo._pso`` module.

    Raises
    ------
    ImportError
        If the extension has not been built for this Python.
    """
    if _extension is None:
        raise ImportError(f"{BUILD_HINT}\nOriginal error: {IMPORT_ERROR}") from IMPORT_ERROR
    return _extension
