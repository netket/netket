from functools import partial
from typing import Callable, Union

import numpy as np

import jax
from jax import numpy as jnp

from flax import linen as nn

from netket import jax as nkjax
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
    AbstractSuperOperator,
    Squared,
)

from .mc_state import MCState
from .mc_mixed_state import MCMixedState

AFunType = Callable[[nn.Module, PyTree, jnp.ndarray], jnp.ndarray]
ATrainFunType = Callable[
    [nn.Module, PyTree, jnp.ndarray, Union[bool, PyTree]], jnp.ndarray
]


def _check_hilbert(A, B):
    if A.hilbert != B.hilbert:
        raise NotImplementedError(  # pragma: no cover
            f"Non matching hilbert spaces {A.hilbert} and {B.hilbert}"
        )


def local_value_kernel(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCState and generic operators
    """
    return jnp.sum(mel * jnp.exp(logpsi(pars, σp) - logpsi(pars, σ)))


def local_value_squared_kernel(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCState and Squared (generic) operators
    """
    return jnp.abs(local_value_kernel(logpsi, pars, σ, σp, mel)) ** 2


def local_value_op_op_cost(logpsi, pars, σ, σp, mel):
    """
    local_value kernel for MCMixedState and generic operators
    """
    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))


@dispatch.multi((MCState, Squared), (MCMixedState, Squared))
def expect(vstate: MCState, Ô: Squared) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    σ = vstate.samples

    σp, mels = Ô.parent.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_squared_kernel,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@dispatch.multi((MCState, AbstractOperator), (MCMixedState, AbstractSuperOperator))
def expect(vstate: MCState, Ô: AbstractOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    σ = vstate.samples

    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_kernel,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@dispatch
def expect(vstate: MCMixedState, Ô: AbstractOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate.diagonal, Ô)

    σ = vstate.diagonal.samples

    σp, mels = Ô.get_conn_padded(np.asarray(σ).reshape((-1, σ.shape[-1])))

    return _expect(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        local_value_op_op_cost,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@partial(jax.jit, static_argnums=(1, 2))
def _expect(
    machine_pow: int,
    model_apply_fun: Callable,
    local_value_kernel: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    σp: jnp.ndarray,
    mels: jnp.ndarray,
) -> Stats:
    σ_shape = σ.shape

    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    local_value_vmap = jax.vmap(
        partial(local_value_kernel, logpsi),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, σ, σp, mels, n_chains=σ_shape[0]
    )

    return Ō_stats
