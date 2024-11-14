# Copyright 2022 The NetKet Authors - All rights reserved.
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

import jax.numpy as jnp
import jax
from functools import partial

from netket.vqs import FullSumState, expect, expect_and_grad
import netket.jax as nkjax
from netket.utils import mpi
from netket.stats import Stats

from .variance_operator import VarianceObservable


@expect.dispatch
def expect(vstate: FullSumState, variance_operator: VarianceObservable):
    if variance_operator.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    operator_mtrx = variance_operator.operator.to_dense()
    operator_squared_mtrx = variance_operator.operator_squared.to_dense()

    return expect_and_grad_inner_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        operator_mtrx,
        operator_squared_mtrx,
        return_grad=False,
    )


@expect_and_grad.dispatch
def expect_and_grad(
    vstate: FullSumState,
    variance_operator: VarianceObservable,
    *,
    mutable,
):
    if variance_operator.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    operator_mtrx = variance_operator.operator.to_dense()
    operator_squared_mtrx = variance_operator.operator_squared.to_dense()

    return expect_and_grad_inner_fs(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        vstate._all_states,
        operator_mtrx,
        operator_squared_mtrx,
        return_grad=True,
    )


@partial(jax.jit, static_argnames=("afun", "return_grad"))
def expect_and_grad_inner_fs(
    afun, params, model_state, sigma, operator_mtrx, operator_squared_mtrx, return_grad
):
    def expect_kernel_var(params):
        W = {"params": params, **model_state}

        state = jnp.exp(afun(W, sigma))
        state = state / jnp.linalg.norm(state)

        O_mean = state.conj() @ (operator_mtrx @ state)
        O2_mean = state.conj() @ (operator_squared_mtrx @ state)

        return O2_mean - O_mean**2

    if not return_grad:
        var = expect_kernel_var(params)
        return Stats(mean=var, error_of_mean=0.0, variance=0.0)

    var, var_vjp_fun = nkjax.vjp(expect_kernel_var, params, conjugate=True)

    var_grad = var_vjp_fun(jnp.ones_like(var))[0]
    var_grad = jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], var_grad)

    return Stats(mean=var, error_of_mean=0.0, variance=0.0), var_grad
