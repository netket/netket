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

import warnings


from netket import jax as nkjax
from netket.stats import Stats
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    DiscreteJaxOperator,
    ContinuousOperator,
    Squared,
)

# to move up once stabilized
from netket.operator._abstract_observable import AbstractObservable

from netket.vqs import expect

from netket.vqs.mc import (
    kernels,
    get_local_kernel,
)

from .state import MCState
from netket.vqs.mc import local_estimators


# Dispatches to select what expect-kernel to use
@dispatch
def get_local_kernel(vstate: MCState, Ô: Squared, chunk_size: int):  # noqa: F811
    return kernels.local_value_squared_kernel_chunked


@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCState, Ô: DiscreteJaxOperator, chunk_size: int
):  # noqa: F811
    return kernels.local_value_kernel_jax_chunked


@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCState, Ô: DiscreteOperator, chunk_size: int
):
    return kernels.local_value_kernel_chunked


def _local_continuous_kernel(logpsi, pars, σ, op, *, chunk_size=None):
    return nkjax.apply_chunked(
        lambda op, x: op._expect_kernel(logpsi, pars, x),
        in_axes=(None, 0),
        chunk_size=chunk_size,
    )(op, σ)


@dispatch
def get_local_kernel(  # noqa: F811
    vstate: MCState, Ô: ContinuousOperator, chunk_size: int
):
    return _local_continuous_kernel


# If batch_size is unspecified, set it to None
@expect.dispatch
def expect_chunking_unspecified(vstate: MCState, operator: AbstractObservable):
    return expect(vstate, operator, None)


# if no implementation exists for batched, fall back to unbatched methods.
@expect.dispatch(precedence=-10)
def expect_fallback(
    vstate: MCState, operator: AbstractObservable, chunk_size: int | tuple
):  # noqa: F811
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_grad method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists."
    )

    return expect(vstate, operator, None)


@expect.dispatch
def expect_mcstate_operator_chunked(
    vstate: MCState, Ô: AbstractOperator, chunk_size: int
) -> Stats:  # noqa: F811
    return local_estimators(vstate, Ô, chunk_size).to_stats()
