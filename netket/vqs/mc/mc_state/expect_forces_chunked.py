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

from functools import partial
from typing import Any
from collections.abc import Callable
import warnings

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.operator._abstract_observable import AbstractObservable
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree

from netket.vqs import expect_and_forces
from netket.vqs.mc import (
    get_local_kernel,
    get_local_kernel_arguments,
)

from .state import MCState


# If batch_size is unspecified, set it to None
@expect_and_forces.dispatch
def expect_and_forces_chunking_unspecified(  # noqa: F811
    vstate: MCState,
    operator: AbstractObservable,
    **kwargs,
):
    return expect_and_forces(vstate, operator, None, **kwargs)


# if no implementation exists for batched, run the code unbatched
@expect_and_forces.dispatch(precedence=-10)
def expect_and_forces_fallback(  # noqa: F811
    vstate: MCState,
    operator: AbstractObservable,
    chunk_size: Any,
    *args,
    **kwargs,
):
    warnings.warn(
        f"Ignoring chunk_size={chunk_size} for expect_and_forces method with signature "
        f"({type(vstate)}, {type(operator)}) because no implementation supporting "
        f"chunking for this signature exists.",
        stacklevel=2,
    )

    return expect_and_forces(vstate, operator, None, *args, **kwargs)


@expect_and_forces.dispatch
def expect_and_forces_impl(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    chunk_size: int,
    *,
    mutable: CollectionFilter = False,
) -> tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    Ō, Ō_grad, new_model_state = forces_expect_hermitian_chunked(
        chunk_size,
        local_estimator_fun,
        vstate._apply_fun,
        mutable,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def forces_expect_hermitian_chunked(
    chunk_size: int,
    local_value_kernel_chunked: Callable,
    model_apply_fun: Callable,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree]:
    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel_chunked(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
        chunk_size=chunk_size,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]))

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    if mutable is False:
        vjp_fun_chunked = nkjax.vjp_chunked(
            lambda w, σ: model_apply_fun({"params": w, **model_state}, σ),
            parameters,
            σ,
            conjugate=True,
            chunk_size=chunk_size,
            chunk_argnums=1,
            nondiff_argnums=1,
        )
        new_model_state = None
    else:
        raise NotImplementedError

    Ō_grad = vjp_fun_chunked(
        (jnp.conjugate(O_loc) / n_samples),
    )[0]

    Ō_grad, _ = mpi.mpi_sum_jax(Ō_grad)

    return Ō, Ō_grad, new_model_state
