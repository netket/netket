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
from typing import Any, Callable, Tuple

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractOperator,
)

from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)

from .state import MCState


@dispatch
def expect_and_forces(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    Ō, Ō_grad, new_model_state = forces_expect_hermitian(
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


@partial(jax.jit, static_argnums=(0, 1, 2))
def forces_expect_hermitian(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    n_samples = σ.shape[0] * mpi.n_nodes

    O_loc = local_value_kernel(
        model_apply_fun,
        {"params": parameters, **model_state},
        σ,
        local_value_args,
    )

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

    # Then compute the vjp.
    # Code is a bit more complex than a standard one because we support
    # mutable state (if it's there)
    is_mutable = mutable is not False
    _, vjp_fun, *new_model_state = nkjax.vjp(
        lambda w: model_apply_fun({"params": w, **model_state}, σ, mutable=mutable),
        parameters,
        conjugate=True,
        has_aux=is_mutable,
    )
    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state
