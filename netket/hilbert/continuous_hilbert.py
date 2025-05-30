"""Deprecated alias to :mod:`netket.experimental.hilbert`."""

from netket.utils.deprecation import warn_deprecation

warn_deprecation(
    "netket.hilbert.ContinuousHilbert is deprecated. Use netket.experimental.hilbert.ContinuousHilbert instead."
)

from netket.experimental.hilbert.continuous_hilbert import *
