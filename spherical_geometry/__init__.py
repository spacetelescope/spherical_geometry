import os

__all__ = ["__version__", "HAS_C_UFUNCS"]

try:
    from ._version import version as __version__
except ImportError:
    __version__ = ''


DISABLE_C_UFUNCS = os.environ.get("DISABLE_SPHR_GEOM_C_UFUNCS", "false") == "true"
if DISABLE_C_UFUNCS:
    HAS_C_UFUNCS = False
else:
    try:
        from . import math_util  # noqa: F401
        HAS_C_UFUNCS = True
    except ImportError:
        HAS_C_UFUNCS = False
