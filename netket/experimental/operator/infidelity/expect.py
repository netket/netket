from functools import partial

import jax
import jax.numpy as jnp

from netket.vqs import MCState, expect, expect_and_grad
import netket.jax as nkjax
from netket.stats import Stats

from .infidelity_operator import InfidelityOperator


@expect.dispatch
def infidelity(vstate: MCState, op: InfidelityOperator, chunk_size: None):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_sampling_inner(
        vstate._apply_fun,
        op.target_state._apply_fun,
        vstate.parameters,
        op.target_state.parameters,
        vstate.model_state,
        op.target_state.model_state,
        vstate.samples,
        op.target_state.samples,
        op.cv_coeff,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: MCState,
    op: InfidelityOperator,
    chunk_size: None,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_sampling_inner(
        vstate._apply_fun,
        op.target_state._apply_fun,
        vstate.parameters,
        op.target_state.parameters,
        vstate.model_state,
        op.target_state.model_state,
        vstate.samples,
        op.target_state.samples,
        op.cv_coeff,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "afun_t", "return_grad"))
def infidelity_sampling_inner(
    afun,
    afun_t,
    params,
    params_t,
    model_state,
    model_state_t,
    sigma,
    sigma_t,
    cv_coeff,
    return_grad,
):
    N = sigma.shape[-1]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)

    def kernel_fun(params_all, samples_all):
        params, params_t = params_all
        σ, σ_t = samples_all

        W = {"params": params, **model_state}
        W_t = {"params": params_t, **model_state_t}

        log_val = afun_t(W_t, σ) - afun(W, σ)
        log_val_t = afun(W, σ_t) - afun_t(W_t, σ_t)

        return log_val, log_val_t

    log_val, log_val_t = kernel_fun((params, params_t), (σ, σ_t))

    res = jnp.exp(log_val + log_val_t).real

    if cv_coeff is not None:
        res = res + cv_coeff * (jnp.exp(2 * (log_val + log_val_t).real) - 1)

    mean = jnp.mean(res)
    variance = jnp.var(res)
    error = jnp.sqrt(variance / res.shape[-1])

    I_stats = Stats(
        mean=1 - mean,
        error_of_mean=error,
        variance=variance,
    )

    if not return_grad:
        return I_stats

    Hloc = jnp.exp(log_val) * jnp.mean(jnp.exp(log_val_t))
    Hloc = Hloc - jnp.mean(Hloc)

    _, Ok_vjp = nkjax.vjp(
        lambda params: afun({"params": params, **model_state}, σ),
        params,
        conjugate=True,
    )

    I_grad = Ok_vjp(Hloc.conj())[0]

    I_grad = jax.tree_util.tree_map(lambda x: x / σ.shape[0], I_grad)

    I_grad = jax.tree_util.tree_map(lambda x: 2 * jnp.real(x), I_grad)

    I_grad = jax.tree_util.tree_map(lambda x: -x, I_grad)

    return I_stats, I_grad
