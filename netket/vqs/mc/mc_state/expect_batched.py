from functools import partial
from typing import Callable

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    DiscreteOperator,
    Squared,
)

from netket.vqs import expect

from netket.vqs.mc import kernels, check_hilbert, get_configs, get_fun

from .state import MCState

# Dispatches to select what expect-kernel to use
@dispatch
def get_fun(vstate: MCState, Ô: Squared, batch_size: int):
    return kernels.local_value_squaredkernel_chunked


@dispatch
def get_fun(vstate: MCState, Ô: DiscreteOperator, batch_size: int):
    return kernels.local_valuekernel_chunked


# If batch_size is None, ignore it and remove it from signature so that we fall back
# to already implemented methods
@expect.dispatch
def expect_nochunking(vstate: MCState, operator: DiscreteOperator, batch_size: None):
    return expect(vstate, operator)


# if no implementation exists for batched, fall back to unbatched methods.
@expect.dispatch
def expect_fallback(vstate: MCState, operator: DiscreteOperator, batch_size):
    if config.FLAGS["NETKET_DEBUG"]:
        print(
            "Ignoring `batch_size={batch_size}` because no implementation supporting:"
            "batching exists."
        )

    return expect(vstate, operator)


@expect.dispatch
def expect_mcstate_operator_batched(
    vstate: MCState, Ô: DiscreteOperator, batch_size: int
) -> Stats:  # noqa: F811
    σ, σp, mels = get_configs(vstate, Ô)

    local_estimator_fun = get_fun(vstate, Ô, batch_size)

    return _expect_chunking(
        vstate._apply_fun,
        local_estimator_fun,
        batch_size,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_chunking(
    model_apply_fun: Callable,
    local_value_kernel: Callable,
    batch_size: int,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    if jnp.ndim(σp) != 3:
        σp = σp.reshape((σ.shape[0], -1, σ_shape[-1]))
        mels = mels.reshape(σp.shape[:-1])

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    local_value_vmap = partial(local_value_kernel, logpsi, batch_size=batch_size)

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, σ, σp, mels, n_chains=σ_shape[0]
    )

    return Ō_stats
