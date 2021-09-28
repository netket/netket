
from plum import dispatch

from plum import dispatch

from netket.utils import struct

import jax.numpy as jnp

from ...base import AbstractAlgorithm
from ...base import AbstractIntegrator

from .base import AbstractODEAlgorithm, AbstractODEAlgorithmCache, get_current_adaptive_order, perform_step

@struct.dataclass
class AbstractODETableauAlgorithm(AbstractODEAlgorithm):
    @property
    def tableau(self):
        raise NotImplementedError

    @property
    def alg_order(self):
        return self.tableau.order[0]

    @property
    def n_stages(self):
        return self.tableau.stages

    @property
    def is_adaptive(self):
        return self.tableau.is_adaptive

    def make_cache(self, u, reltol_internal):
        return TrivialAlgorithmCache()

@struct.dataclass
class TrivialAlgorithmCache(AbstractODEAlgorithmCache):
    pass

@get_current_adaptive_order.dispatch
def get_current_adaptive_order(alg: AbstractODETableauAlgorithm, cache):
    return alg.tableau[0]

@perform_step.dispatch
def perform_step(integrator: AbstractIntegrator, alg: AbstractODETableauAlgorithm, cache: AbstractODEAlgorithmCache, *, repeat_step=False):
    f = integrator.f
    t = integrator.t
    dt = integrator.dt
    u_t = integrator.u

    tableau = alg.tableau

    k = tableau._get_ks(f, t, dt, u_t)

    b = tableau.b[0] if tableau.b.ndim == 2 else tableau.b
    u_tp1 = u_t + dt * b @ k

    if integrator.opts.adaptive:
        if not tableau.is_adaptive:
            raise RuntimeError(f"{self} is not adaptive")

        u_err = ut * (tableau.b[0]-tableau.b[1]) @ k
        # todo update error

    # cache used for interpolation
    integrator.k = k.at[jnp.array([1,-1])].get()
    integrator.u = u_tp1

    return integrator
