# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# The score function (REINFORCE) gradient estimator of an expectation

from functools import partial

import jax
from jax import numpy as jnp

from ._grad import grad
from ._vjp import vjp as nkvjp


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def expect(log_pdf, expected_fun, model, pars, σ, *expected_fun_args):
    L_σ = expected_fun(model, pars, σ, *expected_fun_args)
    return L_σ.mean(axis=0)


def expect_fwd(log_pdf, expected_fun, model, pars, σ, *expected_fun_args):
    L_σ = expected_fun(model, pars, σ, *expected_fun_args)
    L̄_σ = L_σ.mean(axis=0)

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ
    return L̄_σ, (pars, σ, expected_fun_args, ΔL_σ)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the batch dimension and without a loop through the batch dimension
def expect_bwd(log_pdf, expected_fun, model, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals

    def f(pars, σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(model, pars, σ, *cost_args)
        # TODO: use nk.stats.mean, which uses MPI already
        out = (term1 + term2).mean(axis=0)
        out = out.sum()
        return out

    _, pb = nkvjp(f, pars, σ, *cost_args)
    grad_f = pb(dout)
    return grad_f


expect.defvjp(expect_fwd, expect_bwd)
