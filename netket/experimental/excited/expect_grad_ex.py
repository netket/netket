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
from typing import Any, Callable, Tuple, Literal

import jax
from jax import numpy as jnp


from netket import jax as nkjax
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree


from netket.vqs.mc import (
    get_local_kernel_arguments,
    get_local_kernel,
)

from netket.vqs.mc.mc_state.state import MCState, expect_and_grad

from .operator import OperatorWithPenalty
from .expect import penalty_kernel


# compute the loss and gradient with penalty operator and weight input.
@expect_and_grad.dispatch
def expect_and_grad_infidelity_penalty(
    vstate: MCState,
    Ô: OperatorWithPenalty,
    use_covariance: Literal[True],
    *,
    mutable,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô.operator)
    local_estimator_fun = get_local_kernel(vstate, Ô.operator)

    shift_list = Ô.shifts

    # here we make lists to be passed as input.
    σ_list = []
    model_state_list = []
    pars_list = []
    for state_i in Ô.states:
        σ_list.append(state_i.samples)

        model_state_i = state_i.model_state
        model_state_list.append(model_state_i)

        pars_i = state_i.parameters
        pars_list.append(pars_i)

    Ō, Ō_grad, new_model_state = grad_expect_hermitian_ex(
        local_estimator_fun,
        vstate._apply_fun,
        mutable,
        penalty_kernel,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
        σ_list,
        model_state_list,
        pars_list,
        shift_list,
    )

    if mutable is not False:
        vstate.model_state = new_model_state

    return Ō, Ō_grad


@partial(
    jax.jit,
    static_argnames=(
        "local_value_kernel",
        "model_apply_fun",
        "mutable",
        "penalty_kernel",
    ),
)
def grad_expect_hermitian_ex(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    mutable: bool,
    penalty_kernel: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
    σ_list: Any,
    model_state_list: Any,
    pars_list: Any,
    shift_list: Any,
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

    # we need to store O_loc as E_loc in order to modify it afterwards and output loss function in the end.

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)
    O_loc -= Ō.mean

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

    # here we write a loop over the previously determined penalty states, and use their information to modify E_loc and O_grad.
    for i in range(len(shift_list)):
        shift_i = shift_list[i]
        σ_i = σ_list[i]
        pars_i = pars_list[i]
        model_state_i = model_state_list[i]
        σ_i_shape = σ_i.shape
        if jnp.ndim(σ_i) != 2:
            σ_i = σ_i.reshape((-1, σ_i_shape[-1]))

        # psi_loc_1 refers to the one with varying state parameters above
        psi_loc_1 = penalty_kernel(
            model_apply_fun,
            {"params": parameters, **model_state},
            {"params": pars_i, **model_state_i},
            σ_i,
        )
        psi_1 = statistics(psi_loc_1.reshape(σ_i_shape[:-1]).T)

        # psi_loc_2 refers to the one with fixed state parameters above
        psi_loc_2 = penalty_kernel(
            model_apply_fun,
            {"params": pars_i, **model_state_i},
            {"params": parameters, **model_state},
            σ,
        )
        psi_2 = statistics(psi_loc_2.reshape(σ_shape[:-1]).T)

        # we do not modify the cost function here, only display the energy is good enough
        # now we can modify the cost function, based on E_loc
        # E_loc += shift_i * psi_1.mean * psi_2.mean

        # now we need to modify the gradient function, simply add to it should work
        psi_loc_2 -= psi_2.mean
        psi_loc_2 *= shift_i * psi_1.mean
        O_loc += psi_loc_2

    Ō_grad = vjp_fun(jnp.conjugate(O_loc) / n_samples)[0]
    Ō_grad = jax.tree_map(
        lambda x, target: (x if jnp.iscomplexobj(target) else x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    O_with_penalty = statistics(O_loc.reshape(σ_shape[:-1]).T)

    new_model_state = new_model_state[0] if is_mutable else None

    return (
        (Ō, O_with_penalty),
        jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad),
        new_model_state,
    )
