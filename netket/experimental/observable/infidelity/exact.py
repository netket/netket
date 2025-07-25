from functools import partial

import jax
import jax.numpy as jnp

from netket.vqs import FullSumState, expect, expect_and_grad
import netket.jax as nkjax
from netket.stats import Stats

from netket.experimental.observable.infidelity.infidelity_operator import (
    InfidelityOperator,
)


@expect.dispatch
def infidelity(vstate: FullSumState, op: InfidelityOperator):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_fullsum_inner(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.target_state.to_array(),
        vstate._all_states,
        return_grad=False,
    )


@expect_and_grad.dispatch
def infidelity(  # noqa: F811
    vstate: FullSumState,
    op: InfidelityOperator,
    *,
    mutable,
):
    if op.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    return infidelity_fullsum_inner(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        op.target_state.to_array(),
        vstate._all_states,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "return_grad"))
def infidelity_fullsum_inner(
    afun,
    params,
    model_state,
    state_t,
    σ,
    return_grad,
):
    def expect_fun(params, σ):
        W = {"params": params, **model_state}

        psi = jnp.exp(afun(W, σ))
        psi /= jnp.linalg.norm(psi)

        I = 1 - jnp.absolute(jnp.vdot(psi.conj(), state_t)) ** 2

        return I

    if not return_grad:
        I = expect_fun(params, σ)

        I_stats = Stats(
            mean=I,
            error_of_mean=0.0,
            variance=0.0,
        )

        return I_stats

    I, I_vjp_fun = nkjax.vjp(expect_fun, params, σ, conjugate=True)

    I_grad = I_vjp_fun(jnp.ones_like(I))[0]

    I_stats = Stats(
        mean=I,
        error_of_mean=0.0,
        variance=0.0,
    )

    return I_stats, I_grad
