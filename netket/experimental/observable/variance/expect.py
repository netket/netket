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

from netket.vqs import (
    MCState,
    expect,
    expect_and_grad,
    get_local_kernel,
    get_local_kernel_arguments,
)
import netket.jax as nkjax
from netket.utils import mpi

from .variance_operator import VarianceObservable


@expect.dispatch
def expect(vstate: MCState, variance_operator: VarianceObservable, chunk_size: None):
    if variance_operator.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    local_kernel = get_local_kernel(vstate, variance_operator.operator)
    local_kernel2 = get_local_kernel(vstate, variance_operator.operator_squared)

    sigma, args = get_local_kernel_arguments(vstate, variance_operator.operator)
    sigma, args2 = get_local_kernel_arguments(
        vstate, variance_operator.operator_squared
    )

    return expect_and_grad_inner_mc(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        local_kernel,
        local_kernel2,
        sigma,
        args,
        args2,
        return_grad=False,
    )


@expect_and_grad.dispatch
def expect_and_grad(
    vstate: MCState,
    variance_operator: VarianceObservable,
    chunk_size: None,
    *,
    mutable,
):
    if variance_operator.hilbert != vstate.hilbert:
        raise TypeError("Hilbert spaces should match")

    local_kernel = get_local_kernel(vstate, variance_operator.operator)
    local_kernel2 = get_local_kernel(vstate, variance_operator.operator_squared)

    sigma, args = get_local_kernel_arguments(vstate, variance_operator.operator)
    sigma, args2 = get_local_kernel_arguments(
        vstate, variance_operator.operator_squared
    )

    return expect_and_grad_inner_mc(
        vstate._apply_fun,
        vstate.parameters,
        vstate.model_state,
        local_kernel,
        local_kernel2,
        sigma,
        args,
        args2,
        return_grad=True,
    )


@partial(
    jax.jit, static_argnames=("afun", "local_kernel", "local_kernel2", "return_grad")
)
def expect_and_grad_inner_mc(
    afun,
    params,
    model_state,
    local_kernel,
    local_kernel2,
    sigma,
    args,
    args2,
    return_grad,
):
    n_chains = sigma.shape[0]

    σ = sigma.reshape(-1, sigma.shape[-1])

    def expect_kernel_var(params):
        log_pdf = lambda params, σ: 2 * afun({"params": params, **model_state}, σ).real

        def local_kernel_var(params, σ):
            W = {"params": params, **model_state}

            O_mean = nkjax.expect(
                log_pdf,
                lambda params, σ: local_kernel(
                    afun, {"params": params, **model_state}, σ, args
                ),
                params,
                σ,
                n_chains=n_chains,
            )[0]

            O2_loc = local_kernel2(afun, W, σ, args2)

            return (O2_loc - O_mean**2).real

        return nkjax.expect(log_pdf, local_kernel_var, params, σ, n_chains=n_chains)

    if not return_grad:
        var, var_stats = expect_kernel_var(params)
        return var_stats

    var, var_vjp_fun, var_stats = nkjax.vjp(
        expect_kernel_var, params, has_aux=True, conjugate=True
    )

    var_grad = var_vjp_fun(jnp.ones_like(var))[0]
    var_grad = jax.tree_util.tree_map(lambda x: mpi.mpi_sum_jax(x)[0], var_grad)

    return var_stats, var_grad
