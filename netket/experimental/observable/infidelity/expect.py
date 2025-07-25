from functools import partial

import jax
import jax.numpy as jnp

from netket.vqs import MCState, expect
from netket.stats import Stats

from netket.experimental.observable.infidelity.infidelity_operator import (
    InfidelityOperator,
)


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
    )


@partial(jax.jit, static_argnames=("afun", "afun_t"))
def get_kernels(afun, afun_t, params, params_t, σ, σ_t, model_state, model_state_t):
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}

    log_val = afun_t(W_t, σ) - afun(W, σ)
    log_val_t = afun(W, σ_t) - afun_t(W_t, σ_t)

    return log_val, log_val_t


def get_local_estimator(vstate, target_state, cv_coeff=-0.5):
    log_val, log_val_t = get_kernels(
        vstate._apply_fun,
        target_state._apply_fun,
        vstate.parameters,
        target_state.parameters,
        vstate.samples,
        target_state.samples,
        vstate.model_state,
        target_state.model_state,
    )

    Hloc = jnp.exp(log_val) * jnp.mean(jnp.exp(log_val_t))

    Hloc_cv = jnp.exp(log_val + log_val_t).real + cv_coeff * (
        jnp.exp(2 * (log_val + log_val_t).real) - 1
    )

    return Hloc, Hloc_cv


@partial(jax.jit, static_argnames=("afun", "afun_t"))
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
):
    N = sigma.shape[-1]

    σ = sigma.reshape(-1, N)
    σ_t = sigma_t.reshape(-1, N)

    log_val, log_val_t = get_kernels(
        afun, afun_t, params, params_t, σ, σ_t, model_state, model_state_t
    )

    if cv_coeff is not None:
        kernel_vals = jnp.exp(log_val + log_val_t).real + cv_coeff * (
            jnp.exp(2 * (log_val + log_val_t).real) - 1
        )
    else:
        kernel_vals = jnp.exp(log_val) * jnp.mean(jnp.exp(log_val_t))

    mean = jnp.mean(kernel_vals)
    variance = jnp.var(kernel_vals)
    error = jnp.sqrt(variance / kernel_vals.shape[-1])

    I_stats = Stats(
        mean=1 - mean,
        error_of_mean=error,
        variance=variance,
    )

    return I_stats
