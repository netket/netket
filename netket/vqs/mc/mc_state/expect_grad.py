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
def expect_and_grad(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    use_covariance: TrueT,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:
    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    Ō, Ō_grad, new_model_state = grad_expect_hermitian(
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


# pure state, squared operator
@dispatch.multi(
    (MCState, Squared[DiscreteOperator], FalseT),
    (MCState, Squared[AbstractOperator], FalseT),
    (MCState, AbstractOperator, FalseT),
)
def expect_and_grad(  # noqa: F811
    vstate,
    Ô,
    use_covariance,
    *,
    mutable: Any,
) -> Tuple[Stats, PyTree]:

    if not isinstance(Ô, Squared) and not config.FLAGS["NETKET_EXPERIMENTAL"]:
        raise RuntimeError(
            """
            Computing the gradient of non hermitian operator is an
            experimental feature under development and is known not to
            return wrong values sometimes.

            If you want to debug it, set the environment variable
            NETKET_EXPERIMENTAL=1
            """
        )

    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô)

    Ō, Ō_grad, new_model_state = grad_expect_operator_kernel(
        local_estimator_fun,
        vstate._apply_fun,
        vstate.sampler.machine_pow,
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
def grad_expect_hermitian(
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

    Ō_grad = jax.tree_multimap(
        lambda x, target: (x if jnp.iscomplexobj(target) else 2 * x.real).astype(
            target.dtype
        ),
        Ō_grad,
        parameters,
    )

    new_model_state = new_model_state[0] if is_mutable else None

    return Ō, jax.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), new_model_state


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def grad_expect_operator_kernel(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    mutable: bool,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree, Stats]:

    σ_shape = σ.shape
    if jnp.ndim(σ) != 2:
        σ = σ.reshape((-1, σ_shape[-1]))

    is_mutable = mutable is not False
    logpsi = lambda w, σ: model_apply_fun(
        {"params": w, **model_state}, σ, mutable=mutable
    )
    log_pdf = (
        lambda w, σ: machine_pow * model_apply_fun({"params": w, **model_state}, σ).real
    )

    def expect_closure_pars(pars):
        return nkjax.expect(
            log_pdf,
            partial(local_value_kernel, logpsi),
            pars,
            σ,
            local_value_args,
            n_chains=σ_shape[0],
        )

    Ō, Ō_pb, Ō_stats = nkjax.vjp(
        expect_closure_pars, parameters, has_aux=True, conjugate=True
    )
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))[0]

    # This term below is needed otherwise it does not match the value obtained by
    # (ha@ha).collect(). I'm unsure of why it is needed.
    Ō_pars_grad = jax.tree_multimap(
        lambda x, target: x / 2 if jnp.iscomplexobj(target) else x,
        Ō_pars_grad,
        parameters,
    )

    if is_mutable:
        raise NotImplementedError(
            "gradient of non-hermitian operators over mutable models "
            "is not yet implemented."
        )
    new_model_state = None

    return (
        Ō_stats,
        jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], Ō_pars_grad),
        new_model_state,
    )
