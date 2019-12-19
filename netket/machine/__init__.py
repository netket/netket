from __future__ import absolute_import
from .._C_netket.machine import *
from .cxx_machine import *
from .abstract_machine import AbstractMachine

from .py_rbm import *
from .torch import Torch


def _has_jax():
    try:
        import os

        os.environ["JAX_ENABLE_X64"] = "1"
        import jax

        return True
    except ImportError:
        return False


if _has_jax():
    from .jax import *


def MPSPeriodicDiagonal(hilbert, bond_dim, symperiod=-1):
    return MPSPeriodic(hilbert, bond_dim, diag=True, symperiod=symperiod)
