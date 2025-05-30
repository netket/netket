"""Deprecated alias to :mod:`netket.experimental.hilbert`."""

from netket.utils.deprecation import warn_deprecation

warn_deprecation(
    "netket.hilbert.Particle is deprecated. Use netket.experimental.hilbert.Particle instead."
)

from netket.experimental.hilbert.particle import *
