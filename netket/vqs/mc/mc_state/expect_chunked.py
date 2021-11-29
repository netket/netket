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
    AbstractOperator,
    DiscreteOperator,
    ContinuousOperator,
    Squared,
)

from netket.vqs import VariationalState, expect

from netket.vqs.mc import (
    kernels,
    check_hilbert,
    get_local_kernel,
    get_local_kernel_arguments,
)

from .state import MCState

# Dispatches to select what expect-kernel to use
@dispatch
def get_local_kernel(vstate: MCState, Ô: Squared, chunk_size: int):
    return kernels.local_value_squared_kernel_chunked


@dispatch
def get_local_kernel(vstate: MCState, Ô: DiscreteOperator, chunk_size: int):
    return kernels.local_value_kernel_chunked

def _local_continuous_kernel(kernel, logpsi, pars, σ, args, *, chunk_size=None):
    def _kernel(σ):
        return kernel(logpsi, pars, σ, args)
    return nkjax.vmap_chunked(_kernel, in_axes=0, chunk_size=chunk_size)(σ)

@dispatch
def get_local_kernel(vstate: MCState, Ô: ContinuousOperator, chunk_size: int):
    return nkjax.HashablePartial(_local_continuous_kernel, Ô._expect_kernel) 


# If batch_size is None, ignore it and remove it from signature so that we fall back
# to already implemented methods
@expect.dispatch
def expect_nochunking(vstate: MCState, operator: AbstractOperator, chunk_size: None):
    return expect(vstate, operator)


# if no implementation exists for batched, fall back to unbatched methods.
@expect.dispatch
def expect_fallback(vstate: MCState, operator: AbstractOperator, chunk_size):
    if config.FLAGS["NETKET_DEBUG"]:
        print(
            "Ignoring `chunk_size={chunk_size}` because no implementation supporting:"
            "chunking exists."
        )

    return expect(vstate, operator)


@expect.dispatch
def expect_mcstate_operator_chunked(
    vstate: MCState, Ô: AbstractOperator, chunk_size: int
) -> Stats:  # noqa: F811
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    return _expect_chunking(
        chunk_size,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1, 2))
def _expect_chunking(
    chunk_size: int,
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    args: PyTree,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    _, Ō_stats = nkjax.expect(
        log_pdf,
        partial(local_value_kernel, logpsi, chunk_size=chunk_size),
        parameters,
        σ,
        args,
        n_chains=σ_shape[0],
    )

    return Ō_stats
