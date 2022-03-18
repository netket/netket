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
from netket import config
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch, TrueT, FalseT

from netket.operator import (
    AbstractOperator,
    DiscreteOperator,
    Squared,
)

from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)

from .state import MCState


@dispatch
def expect_and_natgrad(  # noqa: F811
    vstate: MCState,
    K: Any,
    solver: Any,
    Ô: AbstractOperator,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)
    local_estimator_fun = get_local_kernel(vstate, Ô)

    Ō, Ō_grad = natgrad_expect_hermitian(
        K,
        solver,
        local_estimator_fun,
        vstate,
        σ,
        args,
    )

    return Ō, Ō_grad


@partial(
    jax.jit,
    static_argnums=(
        0,
        1,
        2,
    ),
)
def natgrad_expect_hermitian(
    ntk: Callable,
    solver: Callable,
    local_value_kernel: Callable,
    vs: MCState,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree]:
    model_apply_fun = vs._apply_fun
    model_state = vs.model_state
    parameters = vs.parameters
    mutable = vs.mutable

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

    K = ntk(vs)

    dw_1, sol_data = K.solve(solver, O_loc)
    Ō_grad = K.project_to(dw_1, vs.parameters)

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad)
