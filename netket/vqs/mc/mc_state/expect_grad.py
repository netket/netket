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
from typing import Callable, Optional

import jax
from jax import numpy as jnp
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

from netket import jax as nkjax
from netket import config
from netket.stats import Stats
from netket.utils import mpi, dispatch
from netket.utils.types import PyTree

from netket.operator import (
    AbstractOperator,
    Squared,
)

from netket.vqs import expect_and_grad, expect_and_forces

from ..common import force_to_grad, get_local_kernel_arguments, get_local_kernel

from .state import MCState


# General implementation checking hermitianity
@expect_and_grad.dispatch
def expect_and_grad_default_formula(
    vstate: MCState,
    Ô: AbstractOperator,
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    if use_covariance is None:
        use_covariance = Ô.is_hermitian

    if use_covariance:
        # Implementation of expect_and_grad for `use_covariance == True` (due to the Literal[True]
        # type in the signature).` This case is equivalent to the composition of the
        # `expect_and_forces` and `force_to_grad` functions.
        # return expect_and_grad_from_covariance(vstate, Ô, *args, mutable=mutable)
        Ō, Ō_grad = expect_and_forces(vstate, Ô, chunk_size, *args, mutable=mutable)
        Ō_grad = force_to_grad(Ō_grad, vstate.parameters)
        return Ō, Ō_grad
    else:
        return expect_and_grad_nonhermitian(
            vstate, Ô, chunk_size, *args, mutable=mutable
        )


# Squared is a special operator...
@expect_and_grad.dispatch
def expect_and_grad_squared_op(
    vstate: MCState,
    Ô: Squared,
    chunk_size: Optional[int],
    *args,
    mutable: CollectionFilter = False,
    use_covariance: Optional[bool] = None,
) -> tuple[Stats, PyTree]:
    if use_covariance is not None:
        raise ValueError(
            "Cannot specify `use_covariance` with Squared[...] operator.\n"
            "This operator must use the same formula as non-hermitian operators to work."
        )
    return expect_and_grad_nonhermitian(vstate, Ô, chunk_size, *args, mutable=mutable)


@dispatch.dispatch
def expect_and_grad_nonhermitian(
    vstate: MCState,
    Ô,
    chunk_size: None,
    *,
    mutable: CollectionFilter = False,
):
    if not isinstance(Ô, Squared) and not config.netket_experimental:
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

    Ō, Ō_grad, new_model_state = _grad_expect_nonherm_kernel(
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


@partial(jax.jit, static_argnums=(0, 1, 2, 3))
def _grad_expect_nonherm_kernel(
    local_value_kernel: Callable,
    model_apply_fun: Callable,
    machine_pow: int,
    mutable: CollectionFilter,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> tuple[PyTree, PyTree, Stats]:
    n_chains = σ.shape[0]
    if σ.ndim >= 3:
        σ = jax.lax.collapse(σ, 0, 2)

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
            n_chains=n_chains,
        )

    Ō, Ō_pb, Ō_stats = nkjax.vjp(
        expect_closure_pars, parameters, has_aux=True, conjugate=True
    )
    Ō_pars_grad = Ō_pb(jnp.ones_like(Ō))[0]

    if is_mutable:
        raise NotImplementedError(
            "gradient of non-hermitian operators over mutable models "
            "is not yet implemented."
        )
    new_model_state = None

    return (
        Ō_stats,
        jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], Ō_pars_grad),
        new_model_state,
    )
