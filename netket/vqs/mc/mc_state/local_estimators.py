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
from functools import partial

import jax

from netket import jax as nkjax
from netket.utils.dispatch import dispatch
from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable
from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
    local_estimators,
)
from netket._src.stats.local_estimators import LocalEstimators

from .state import MCState


@partial(jax.jit, static_argnames=("kernel", "apply_fun", "shape"))
def _local_estimators_impl(kernel, apply_fun, shape, variables, samples, extra_args):
    samples = jax.lax.collapse(samples, 0, samples.ndim - 1)
    O_loc = kernel(apply_fun, variables, samples, extra_args)
    return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[1:]), O_loc)


@dispatch
def local_estimators(  # noqa: F811
    vstate: MCState, Ô: AbstractOperator, chunk_size: None
) -> LocalEstimators:
    σ, args = get_local_kernel_arguments(vstate, Ô)
    data = _local_estimators_impl(
        get_local_kernel(vstate, Ô, None),
        vstate._apply_fun,
        σ.shape[:-1],
        vstate.variables,
        σ,
        args,
    )
    return LocalEstimators(data)


@dispatch
def local_estimators(  # noqa: F811
    vstate: MCState, Ô: AbstractOperator, chunk_size: int
) -> LocalEstimators:
    σ, args = get_local_kernel_arguments(vstate, Ô)
    kernel = nkjax.HashablePartial(
        get_local_kernel(vstate, Ô, chunk_size), chunk_size=chunk_size
    )
    data = _local_estimators_impl(
        kernel, vstate._apply_fun, σ.shape[:-1], vstate.variables, σ, args
    )
    return LocalEstimators(data)


# Fallback: no chunked implementation → warn and drop to unchunked
@local_estimators.dispatch(precedence=-10)
def local_estimators_fallback(  # noqa: F811
    vstate: MCState, op: AbstractObservable, chunk_size: int | tuple
) -> LocalEstimators:
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for local_estimators with signature "
        f"({type(vstate)}, {type(op)}) because no implementation supporting "
        f"chunking for this signature exists.",
        stacklevel=2,
    )
    return local_estimators(vstate, op, None)
