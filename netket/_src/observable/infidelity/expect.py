from functools import partial

import jax
import jax.numpy as jnp

import netket.jax as nkjax
from netket.vqs import MCState, expect
from netket.vqs.mc.common import local_estimators
from netket._src.stats.local_estimators import LocalEstimators

from netket._src.observable.infidelity.infidelity_operator import (
    InfidelityOperator,
)


@partial(jax.jit, static_argnames=("afun", "afun_t", "chunk_size"))
def _local_fidelity_estimators(
    afun,
    afun_t,
    params,
    params_t,
    sigma,
    sigma_t,
    model_state,
    model_state_t,
    *,
    weights_t=None,
    cv_coeff=None,
    chunk_size=None,
):
    W = {"params": params, **model_state}
    W_t = {"params": params_t, **model_state_t}

    log_val = nkjax.apply_chunked(
        lambda σ: afun_t(W_t, σ) - afun(W, σ), chunk_size=chunk_size
    )(sigma)
    log_val_t = nkjax.apply_chunked(
        lambda σ_t: afun(W, σ_t) - afun_t(W_t, σ_t), chunk_size=chunk_size
    )(sigma_t)

    if weights_t is None:
        hloc = jnp.exp(log_val) * jnp.mean(jnp.exp(log_val_t))
    else:
        hloc = jnp.exp(log_val) * jnp.sum(weights_t * jnp.exp(log_val_t))

    if cv_coeff is not None:
        hloc_cv = jnp.exp(log_val + log_val_t).real + cv_coeff * (
            jnp.exp(2 * (log_val + log_val_t).real) - 1
        )
    else:
        hloc_cv = hloc

    return hloc, hloc_cv


@local_estimators.dispatch
def _(
    vstate: MCState, op: InfidelityOperator, chunk_size: int | None
) -> LocalEstimators:  # noqa: F811
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    n_chains = vstate.samples.shape[0]
    N = vstate.hilbert.size
    sigma = vstate.samples.reshape(-1, N)
    sigma_t = op.target_state.samples.reshape(-1, N)

    _, fidelity_loc = _local_fidelity_estimators(
        vstate._apply_fun,
        op.target_state._apply_fun,
        vstate.parameters,
        op.target_state.parameters,
        sigma,
        sigma_t,
        vstate.model_state,
        op.target_state.model_state,
        cv_coeff=op.cv_coeff,
        chunk_size=chunk_size,
    )

    # Infidelity = 1 - fidelity; store directly so the scalar path is used.
    data = (1.0 - fidelity_loc).reshape(n_chains, -1)
    return LocalEstimators(data=data)


@expect.dispatch
def infidelity(
    vstate: MCState, op: InfidelityOperator, chunk_size: int | None
):  # noqa: F811
    return local_estimators(vstate, op, chunk_size).to_stats()


def get_local_estimator(
    vstate, target_state, samples, weights, samples_t, weights_t, cv_coeff=-0.5
):
    Hloc, Hloc_cv = _local_fidelity_estimators(
        vstate._apply_fun,
        target_state._apply_fun,
        vstate.parameters,
        target_state.parameters,
        samples,
        samples_t,
        vstate.model_state,
        target_state.model_state,
        weights_t=weights_t,
        cv_coeff=(
            cv_coeff
            if isinstance(target_state, MCState) and isinstance(vstate, MCState)
            else None
        ),
    )

    return Hloc, Hloc_cv
