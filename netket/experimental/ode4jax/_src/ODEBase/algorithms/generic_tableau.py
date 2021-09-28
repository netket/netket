
from plum import dispatch

from plum import dispatch

from netket.utils import struct

import jax.numpy as jnp

from ...base import AbstractAlgorithm
from ...base import AbstractIntegrator

from .base import AbstractODEAlgorithm, AbstractODEAlgorithmCache, get_current_adaptive_order, perform_step
from ..utils import expand_dim

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

    @property
    def is_fsal(self):
        # even if the tableau is not, the generic implementation is fsal
        return True

    def uses_uprev(self, adaptive):
        return True

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
    p = integrator.p
    t = integrator.t
    dt = integrator.dt
    u_t = integrator.u

    tableau = alg.tableau

    ## Compute k
    times = t + tableau.c * dt
    
    k = expand_dim(u_t, tableau.stages)

    # First one (cached)
    k = k.at[0].set(integrator.fsalfirst)

    # Middle ones
    for l in range(1, tableau.stages):
        du_l = tableau.a[l,0:l] @ k.at[0:l].get()
        k_l = f(u_t + dt * du_l, p, times[l], stage=l)
        k = k.at[l].set(k_l)

    # the tableau might not be fsal, but we make the solver
    # behave as a fsal solver for consistency among
    # RK solvers
    if not tableau.is_FSAL:
        integrator.fsallast = f(u_t, p, t+dt)
    else:
        # Cache the last one
        integrator.fsallast = k_l
        #integrator.fsallast = k.at[-1].get()

    ## Compute the updates
    b = tableau.b[0] if tableau.b.ndim == 2 else tableau.b
    u_tp1 = u_t + dt * b @ k

    ## compute error estimates
    if integrator.opts.adaptive:
        if not tableau.is_adaptive:
            raise RuntimeError(f"{self} is not adaptive")

        u_err = dt * (tableau.b[0]-tableau.b[1]) @ k
        integrator.EEst = integrator.opts.errornorm(u_err, u_t, u_tp1, integrator.opts.abstol, integrator.opts.reltol, 
                                                    integrator.opts.internalnorm, t)

    # cache used for interpolation
    integrator.k = k.at[jnp.array([0,-1])].get()
    integrator.u = u_tp1

    return integrator
