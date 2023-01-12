import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from flax import struct

from .base import MetropolisRule
import netket.jax as nkjax


@struct.dataclass
class LangevinRule(MetropolisRule):
    r"""
    A transition rule that uses Langevin dynamics to update samples.
    """

    dt: float = 0.001
    """
    Time step in the Langevin dynamics
    """
    chunk_size: int = None
    """
    Chunk size for computing gradients of the ansatz
    """

    def transition(rule, sampler, machine, parameters, state, key, r):
        if jnp.issubdtype(r.dtype, jnp.complexfloating):
            raise TypeError("LangevinRule does not work with complex basis elements.")

        n_chains = r.shape[0]
        hilb = sampler.hilbert

        pbc = np.array(hilb.n_particles * hilb.pbc, dtype=r.dtype)
        boundary = np.tile(pbc, (n_chains, 1))

        Ls = np.array(hilb.n_particles * hilb.extent, dtype=r.dtype)
        modulus = np.where(np.equal(pbc, False), jnp.inf, Ls)

        # one langevin step
        rp, log_corr = _langevin_step(
            key,
            r,
            machine.apply,
            parameters,
            sampler.machine_pow,
            rule.dt,
            chunk_size=rule.chunk_size,
            return_log_corr=True,
        )

        rp = jnp.where(np.equal(boundary, False), rp, rp % modulus)

        return rp, log_corr

    def __repr__(self):
        return "LangevinRule(dt={})".format(self.dt)


@partial(jax.jit, static_argnames=("axis",))
def _norm_sqr(x, axis=None):
    return jnp.sum(x**2, axis=axis)


@partial(jax.jit, static_argnames=("apply_fun", "chunk_size", "return_log_corr"))
def _langevin_step(
    key,
    r,
    apply_fun,
    parameters,
    machine_pow,
    dt,
    chunk_size=None,
    return_log_corr=True,
):
    """Single step of samples with Langevin dynamics"""

    n_chains, hilb_size = r.shape

    # steps with Langevin dynamics
    noise_vec = jax.random.normal(key, shape=(n_chains, hilb_size), dtype=r.dtype)

    def _log_prob(x):
        """Conversion to a log probability"""
        return machine_pow * apply_fun(parameters, x).real

    def _single_grad(x):
        """Derivative of log_prob with respect to a single sample x"""
        g = nkjax.grad(_log_prob)(x)
        return g if jnp.iscomplexobj(r) else g.real

    grad_logp_r = nkjax.vmap_chunked(_single_grad, chunk_size=chunk_size)(r)

    rp = r + dt * grad_logp_r + jnp.sqrt(2 * dt) * noise_vec

    if not return_log_corr:
        return rp
    else:
        log_q_xp = -0.5 * _norm_sqr(noise_vec, axis=-1)
        grad_logp_rp = nkjax.vmap_chunked(_single_grad, chunk_size=chunk_size)(rp)
        log_q_x = -_norm_sqr(r - rp - dt * grad_logp_rp, axis=-1) / (4 * dt)

        return rp, log_q_x - log_q_xp
