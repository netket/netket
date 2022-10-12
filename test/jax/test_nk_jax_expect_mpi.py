import pytest

import netket as nk
import jax.numpy as jnp
from functools import partial
from typing import Callable, Tuple
from functools import partial
import jax
from jax import numpy as jnp
from netket.utils.types import PyTree
from netket.jax._vjp import vjp as nkvjp
from ..common import netket_disable_mpi
from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
import numpy as np


def expval_grad(vstate, op):

    sigma, args = nk.vqs.get_local_kernel_arguments(vstate, op)
    e_loc = nk.vqs.get_local_kernel(vstate, op)

    return expval_grad_inner(
        vstate._apply_fun, e_loc, vstate.parameters, vstate.model_state, sigma, args
    )


@partial(jax.jit, static_argnames=("apply_fun", "e_loc"))
def expval_grad_inner(apply_fun, e_loc, params, model_state, sigma, args):

    N = sigma.shape[-1]
    n_chains = sigma.shape[1]
    sigma = sigma.reshape(-1, N)

    def expval_pars(params):

        e_loc_ = lambda params_, sigma_: e_loc(
            apply_fun, {"params": params_, **model_state}, sigma_, args
        )

        logpdf = lambda params_, sigma_: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun({"params": params_}, sigma_))))
        )

        return nk.jax.expect(logpdf, e_loc_, params, sigma)[0]

    Ē, E_vjp = nk.jax.vjp(expval_pars, params)
    E_grad = E_vjp(jnp.ones_like(Ē))[0]
    E_grad = jax.tree_map(lambda x: nk.utils.mpi.mpi_mean_jax(x)[0], E_grad)

    return Ē, E_grad


def test_expect_grad_mpi():
    N = 10
    hi = nk.hilbert.Spin(0.5, N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
    H = nk.operator.Ising(hi, g, h=2, J=-1.0)
    model = nk.models.RBM(alpha=1, param_dtype=complex)

    n_samples = 320 * nk.utils.mpi.n_nodes
    r = nk.utils.mpi.rank

    with netket_disable_mpi():
        sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
        vstate = nk.vqs.MCState(
            sampler=sampler, model=model, n_samples=n_samples, seed=1234
        )
        vstate.n_samples = n_samples
        samples = vstate.sample()

        expval_no_mpi, grad_no_mpi = expval_grad(vstate, H)

    nc = samples.shape[1] // nk.utils.mpi.n_nodes
    samples_rank = samples[:, r * nc : (r + 1) * nc, :]
    vstate._samples = samples_rank

    expval_mpi, grad_mpi = expval_grad(vstate, H)

    np.testing.assert_allclose(expval_no_mpi, expval_mpi)

    jax.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y),
        grad_no_mpi,
        grad_mpi,
    )
