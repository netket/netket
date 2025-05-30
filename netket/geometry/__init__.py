"""Geometry classes for continuous Hilbert spaces (deprecated)."""

from netket.experimental.geometry import Cell, FreeSpace

from netket.utils.deprecation import warn_deprecation

warn_deprecation(
    "netket.geometry is deprecated. Use netket.experimental.geometry instead."
)

__all__ = [
    "Cell",
    "FreeSpace",
]
