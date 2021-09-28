
from plum import dispatch

from plum import dispatch

from netket.utils import struct

import jax.numpy as jnp

from ...base import AbstractAlgorithm
from ...base import AbstractIntegrator

from .generic_tableau import AbstractODETableauAlgorithm, TrivialAlgorithmCache, get_current_adaptive_order

from . import tableau


@struct.dataclass
class AbstractODERKAlgorithm(AbstractODETableauAlgorithm):
    pass

@struct.dataclass
class Euler(AbstractODERKAlgorithm):
    
    @property
    def tableau(self):
        return tableau.bt_feuler

@struct.dataclass
class RK4(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk4

@struct.dataclass
class RK23(AbstractODERKAlgorithm):
    @property
    def tableau(self):
        return tableau.bt_rk23

