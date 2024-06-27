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

from typing import Callable
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import Stats, statistics as mpi_statistics
from netket.utils import HashablePartial
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
    ContinuousOperator,
    DiscreteJaxOperator,
)

from netket.vqs.mc import (
    kernels,
    check_hilbert,
    get_local_kernel_arguments,
    get_local_kernel,
)

from .state import MCState


@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: Squared):  # noqa: F811
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, (σp, mels)


@dispatch
def get_local_kernel(vstate: MCState, Ô: Squared):  # noqa: F811
    return kernels.local_value_squared_kernel


@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: DiscreteOperator):  # noqa: F811
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.get_conn_padded(σ)
    return σ, (σp, mels)


@dispatch
def get_local_kernel(vstate: MCState, Ô: DiscreteOperator):  # noqa: F811
    return kernels.local_value_kernel


@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: DiscreteJaxOperator):  # noqa: F811
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    return σ, Ô


@dispatch
def get_local_kernel(vstate: MCState, Ô: DiscreteJaxOperator):  # noqa: F811
    return kernels.local_value_kernel_jax


@dispatch
def get_local_kernel_arguments(vstate: MCState, Ô: ContinuousOperator):  # noqa: F811
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    return σ, Ô


@dispatch
def get_local_kernel(vstate: MCState, _: ContinuousOperator):  # noqa: F811
    # TODO: this should be moved other to dispatch in order to support MCMixedState
    def _local_kernel_continuous(logpsi, parameters, σ, Ô):
        return Ô._expect_kernel(logpsi, parameters, σ)

    return HashablePartial(_local_kernel_continuous)


# Standard implementation of expect for an MCState (pure) and a generic operator
# The dispatch rule is not strictly needed, as everything currently implemented
# in NetKet only defines a custom get_local_kernel_arguments and get_local_kernel
# but if somebody wants to override behaviour for an existing operator or define
# a completely arbitrary novel type of operator, this makes it much easier.
@dispatch
def expect(
    vstate: MCState, Ô: AbstractOperator, chunk_size: None
) -> Stats:  # noqa: F811
    σ, args = get_local_kernel_arguments(vstate, Ô)
    local_estimator_fun = get_local_kernel(vstate, Ô)

    return _expect(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )


@partial(jax.jit, static_argnums=(0, 1))
def _expect(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Stats:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

    def logpsi(w, σ):
        return model_apply_fun({"params": w, **model_state}, σ)

    def log_pdf(w, σ):
        return machine_pow * model_apply_fun({"params": w, **model_state}, σ).real

    # TODO: Broken until google/jax#11916 is resolved.
    # should uncomment and remove code below once this is fixed
    # _, Ō_stats = nkjax.expect(
    #    log_pdf,
    #    partial(local_value_kernel, logpsi),
    #    parameters,
    #    σ,
    #    local_value_args,
    #    n_chains=n_chains,
    # )

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)
    Ō_stats = mpi_statistics(L_σ.reshape((n_chains, -1)))

    return Ō_stats
