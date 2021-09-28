
from plum import dispatch

from netket.utils import struct

import jax.numpy as jnp

from ...base import AbstractAlgorithm
from ...base import AbstractIntegrator

@struct.dataclass
class AbstractODEAlgorithm(AbstractAlgorithm):

    @property
    def is_adaptive(self):
        return False

    @property
    def alg_order(self):
        raise NotImplementedError

    def make_cache(self, u, reltol_internal):
        raise NotImplementedError

@struct.dataclass
class AbstractODEAlgorithmCache:
    pass

@dispatch
def get_current_adaptive_order(alg: AbstractODEAlgorithm, cache):
    pass

@dispatch.abstract
def perform_step(integrator: AbstractIntegrator, alg: AbstractODEAlgorithm, cache: AbstractODEAlgorithmCache, *, repeat_step=False):
    """
    perform an integration step
    """