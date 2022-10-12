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


def expect(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
) -> Tuple[jnp.ndarray, Stats]:

    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """

    return _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args)


@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ.T)
    # L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ


def _expect_fwd(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean
    # L̄_σ = L_σ.mean(axis=0)

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars, σ, expected_fun_args, ΔL_σ)


def _expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    def f(pars, σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars, σ, *cost_args)
        out = term1 + term2
        out = out.mean()

        return out

    _, pb = nkvjp(f, pars, σ, *cost_args)

    grad_f = pb(jnp.ones_like(_))

    return grad_f


_expect.defvjp(_expect_fwd, _expect_bwd)


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


def expval_grad_new(vstate, op):

    sigma, args = nk.vqs.get_local_kernel_arguments(vstate, op)
    e_loc = nk.vqs.get_local_kernel(vstate, op)

    return expval_grad_new_inner(
        vstate._apply_fun, e_loc, vstate.parameters, vstate.model_state, sigma, args
    )


@partial(jax.jit, static_argnames=("apply_fun", "e_loc"))
def expval_grad_new_inner(apply_fun, e_loc, params, model_state, sigma, args):

    N = sigma.shape[-1]
    n_chains = sigma.shape[1]
    sigma = sigma.reshape(-1, N)

    def expval_new_pars(params):

        e_loc_ = lambda params_, sigma_: e_loc(
            apply_fun, {"params": params_, **model_state}, sigma_, args
        )

        logpdf = lambda params_, sigma_: jnp.log(
            jnp.square(jnp.absolute(jnp.exp(apply_fun({"params": params_}, sigma_))))
        )

        return expect(logpdf, e_loc_, params, sigma)[0]

    Ē, E_vjp = nk.jax.vjp(expval_new_pars, params)
    E_grad = E_vjp(jnp.ones_like(Ē))[0]
    E_grad = jax.tree_map(lambda x: nk.utils.mpi.mpi_mean_jax(x)[0], E_grad)

    return Ē, E_grad


def test_expect_grad_mpi():
    N = 10
    hi = nk.hilbert.Spin(0.5, N)
    g = nk.graph.Hypercube(length=N, n_dim=1, pbc=True)
    H = nk.operator.Ising(hi, g, h=2, J=-1.0)
    model = nk.models.RBM(alpha=1, param_dtype=complex)

    n_samples = 1008
    r = nk.utils.mpi.rank

    with netket_disable_mpi():
        sampler = nk.sampler.MetropolisLocal(hilbert=hi, n_chains=16)
        vstate = nk.vqs.MCState(
            sampler=sampler, model=model, n_samples=n_samples, seed=1234
        )
        vstate.n_samples = n_samples
        samples = vstate.sample()

        expval_no_mpi, grad_no_mpi = expval_grad(vstate, H)
        expval_no_mpi_new, grad_no_mpi_new = expval_grad_new(vstate, H)

    nc = samples.shape[1] // nk.utils.mpi.n_nodes
    samples_rank = samples[:, r * nc : (r + 1) * nc, :]
    vstate._samples = samples_rank

    expval_mpi, grad_mpi = expval_grad(vstate, H)
    expval_mpi_new, grad_mpi_new = expval_grad_new(vstate, H)

    np.testing.assert_allclose(expval_no_mpi, expval_mpi)
    np.testing.assert_allclose(expval_no_mpi_new, expval_mpi_new)

    jax.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y),
        grad_no_mpi_new,
        grad_mpi_new,
    )

    jax.tree_map(
        lambda x, y: np.testing.assert_allclose(x, y),
        grad_no_mpi,
        grad_mpi,
    )
