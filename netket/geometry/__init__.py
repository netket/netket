"""Deprecated geometry classes, moved to :mod:`netket.experimental.geometry`."""

from netket.utils.deprecation import deprecation_getattr

from netket.experimental.geometry import Cell, FreeSpace

_deprecations = {
    "Cell": (
        "netket.geometry.Cell is deprecated: use netket.experimental.geometry.Cell",
        Cell,
    ),
    "FreeSpace": (
        "netket.geometry.FreeSpace is deprecated: use netket.experimental.geometry.FreeSpace",
        FreeSpace,
    ),
}

__getattr__ = deprecation_getattr(__name__, _deprecations)
__all__ = ["Cell", "FreeSpace"]
