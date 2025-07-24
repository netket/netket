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

from collections.abc import Callable
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import Stats, statistics
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch
from netket.utils import HashablePartial

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
    ContinuousOperator,
    DiscreteJaxOperator,
)
from netket.operator._sum import SumGenericOperator
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
    return vstate.samples, Ô


@dispatch
def get_local_kernel(vstate: MCState, Ô: ContinuousOperator):  # noqa: F811
    # TODO: this should be moved other to dispatch in order to support MCMixedState
    return HashablePartial(
        lambda logpsi, params, x, op: op._expect_kernel(logpsi, params, x)
    )


## sum operators


@partial(get_local_kernel_arguments.dispatch, precedence=-2)
def get_local_kernel_args_sum(vs: MCState, O: SumGenericOperator):
    samples = vs.samples
    args = [get_local_kernel_arguments(vs, o)[1] for o in O.operators]
    return samples, (O.coefficients, args)


@partial(get_local_kernel.dispatch, precedence=-2)
def get_local_kernel_sum(vs: MCState, O: SumGenericOperator):
    O_kernels = tuple(get_local_kernel(vs, o) for o in O.operators)

    def sum_eval(logpsi, pars, σ, args, O_kernels):
        coeffs, O_args = args
        O_locs = 0
        for O_kernel, O_arg, k in zip(O_kernels, O_args, coeffs):
            O_locs = O_locs + k * O_kernel(logpsi, pars, σ, O_arg)
        return O_locs

    return HashablePartial(sum_eval, O_kernels=O_kernels)


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

    L_σ = local_value_kernel(logpsi, parameters, σ, local_value_args)
    Ō_stats = statistics(L_σ.reshape((n_chains, -1)))

    return Ō_stats
