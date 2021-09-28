
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

    def uses_uprev(self, adaptive):
        """
        If this algorithm does not use `integrator.uprev` returns False.
        Used to save memory.
        """
        return True

    @property
    def is_fsal(self):
        return False

    @property
    def qmin_default(self):
        if self.is_adaptive:
            return 1/5
        else:
            return 0

    @property
    def qmax_default(self):
        return 10

    @property
    def qsteady_min_default(self):
        return 1

    @property
    def qsteady_max_default(self):
        return 1
    @property
    def gamma_default(self):
        if self.is_adaptive:
            return 9/10
        else:
            return 0

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