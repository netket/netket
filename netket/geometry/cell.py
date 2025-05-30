"""Deprecated geometry module forwarding to :mod:`netket.experimental.geometry`."""

from netket.utils.deprecation import warn_deprecation

warn_deprecation(
    "netket.geometry.cell is deprecated. Use netket.experimental.geometry instead."
)

from netket.experimental.geometry.cell import *
