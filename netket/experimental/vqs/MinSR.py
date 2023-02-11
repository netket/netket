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
import warnings

import jax
from jax import numpy as jnp
from jax import tree_map
from flax.core.scope import CollectionFilter

from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.stats import Stats, statistics
from netket.utils import mpi
from netket.utils.types import PyTree

from netket.vqs import expect_and_forces
from netket.vqs.mc import (
    get_local_kernel,
    get_local_kernel_arguments,
)

from netket.jax._jacobian.neural_tangent_kernel import NeuralTangentKernel
from netket.vqs import MCState


def expect_and_MinSR(  # noqa: F811
    vstate: MCState,
    Ô: AbstractOperator,
    chunk_size: int,
    jacobian_chunk_size: int,
    r_cond: float = 1e-12,
) -> Tuple[Stats, PyTree]:

    """Calculates the expectation value of an operator and the gradient update
    according to the MinSR algorithm in `Chen et al. <https://arxiv.org/pdf/2302.01941.pdf>`

    Args:
        vstate: the variational state
        Ô : a hermitian operator
        chunk_size : An integer over which expect and the final VJP are chunked
        jacobian_chunk_size : An integer over which expect and the final VJP are chunked

    Returns:
        A tuple containing the expectation value of Ô and the update according to MinSR
    """

    σ, args = get_local_kernel_arguments(vstate, Ô)

    local_estimator_fun = get_local_kernel(vstate, Ô, chunk_size)

    Ō, Ō_grad, new_model_state = expect_and_MinSR_chunked(
        chunk_size,
        jacobian_chunk_size,
        r_cond,
        local_estimator_fun,
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        σ,
        args,
    )

    return Ō, Ō_grad


# @partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def expect_and_MinSR_chunked(
    chunk_size: int,
    jcs: int,
    r_cond: float,
    local_value_kernel_chunked: Callable,
    model_apply_fun: Callable,
    parameters: PyTree,
    model_state: PyTree,
    σ: jnp.ndarray,
    local_value_args: PyTree,
) -> Tuple[PyTree, PyTree]:

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

    Ō = statistics(O_loc.reshape(σ_shape[:-1]).T)

    O_loc -= Ō.mean

    def grad(w, σ):

        return model_apply_fun({"params": w, **model_state}, σ) / n_samples

    vjp_fun_chunked = nkjax.vjp_chunked(
        grad,
        parameters,
        σ,
        conjugate=False,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    grad_O_mean = vjp_fun_chunked(
        jnp.ones_like(O_loc),
    )[0]

    NTK = jnp.zeros([n_samples, n_samples], dtype="complex")

    for i in range(n_samples // jcs):
        for j in range(n_samples // jcs):
            NTK = NTK.at[i * jcs : (i + 1) * jcs, j * jcs : (j + 1) * jcs].set(
                NeuralTangentKernel(
                    model_apply_fun,
                    parameters,
                    grad_O_mean,
                    σ[i * jcs : (i + 1) * jcs],
                    σ[j * jcs : (j + 1) * jcs],
                    "complex",
                )
            )

    NTK = jnp.linalg.pinv(NTK / n_samples, rcond=r_cond, hermitian=True)

    O_loc = jnp.matmul(NTK, jnp.conj(O_loc))

    def centered_apply(w, σ):
        out = model_apply_fun({"params": w, **model_state}, σ)

        return out - jnp.mean(out)

    vjp_fun_chunked = nkjax.vjp_chunked(
        centered_apply,
        parameters,
        σ,
        conjugate=False,
        chunk_size=chunk_size,
        chunk_argnums=1,
        nondiff_argnums=1,
    )

    Ō_grad = vjp_fun_chunked(
        O_loc / n_samples,
    )[0]

    return Ō, tree_map(lambda x: mpi.mpi_sum_jax(x)[0], Ō_grad), None
