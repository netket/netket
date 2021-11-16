# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Union
from functools import partial

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    DiscreteOperator,
    AbstractSuperOperator,
    Squared,
    ContinuousOperator,
)

from netket.vqs.mc import kernels, check_hilbert, get_configs, get_fun

from .state import MCState


@dispatch
def get_configs(vstate: MCState, Ô: Squared):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, σp, mels


@dispatch
def get_fun(vstate: MCState, Ô: Squared):
    return kernels.local_value_squared_kernel


@dispatch
def get_configs(vstate: MCState, Ô: DiscreteOperator):
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, σp, mels


@dispatch
def get_fun(vstate: MCState, Ô: DiscreteOperator):
    return kernels.local_value_kernel


@dispatch
def expect(vstate: MCState, Ô: Union[DiscreteOperator,Squared[DiscreteOperator]]) -> Stats:  # noqa: F811
    # Standard implementation of expect for an MCState (pure) and a generic Discrete
    # Operator
    σ, σp, mels = get_configs(vstate, Ô)

    local_estimator_fun = get_fun(vstate, Ô)

    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        σp,
        mels,
    )


@dispatch
def expect(vstate: MCState, Ô: ContinuousOperator) -> Stats:  # noqa: F811
    _check_hilbert(vstate, Ô)

    x = vstate.samples
    kernel = Ô._expect_kernel
    return _expect_continuous(
        vstate.sampler.machine_pow,
        vstate._apply_fun,
        kernel,
        vstate.parameters,
        Ô._pack_arguments(),
        vstate.model_state,
        x,
    )


@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
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

    local_value_vmap = jax.vmap(
        partial(local_value_kernel, logpsi),
        in_axes=(None, 0, 0, 0),
        out_axes=0,
    )

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, σ, σp, mels, n_chains=σ_shape[0]
    )

    return Ō_stats


@partial(jax.jit, static_argnums=(1, 2))
def _expect_continuous(
    machine_pow: int,
    model_apply_fun: Callable,
    kernel: Callable,
    parameters: PyTree,
    additional_data: PyTree,
    model_state: PyTree,
    x: jnp.ndarray,
) -> Stats:
    x_shape = x.shape

    if jnp.ndim(x) != 2:
        x = x.reshape((-1, x_shape[-1]))

    def logpsi(w, x):
        return model_apply_fun({"params": w, **model_state}, x)

    log_pdf = lambda w, x: machine_pow * model_apply_fun({"params": w}, x).real

    local_value_vmap = jax.vmap(
        partial(kernel, logpsi),
        in_axes=(None, 0, None),
        out_axes=0,
    )

    _, Ō_stats = nkjax.expect(
        log_pdf, local_value_vmap, parameters, x, additional_data, n_chains=x_shape[0]
    )

    return Ō_stats
