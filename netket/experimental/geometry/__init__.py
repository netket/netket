"""Geometry classes for continuous Hilbert spaces."""

from .cell import Cell, FreeSpace

__all__ = [
    "Cell",
    "FreeSpace",
]

from netket.utils import _hide_submodules

_hide_submodules(__name__)
