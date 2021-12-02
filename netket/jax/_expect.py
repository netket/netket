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

# The score function (REINFORCE) gradient estimator of an expectation

from typing import Callable, Tuple
from functools import partial

import jax
from jax import numpy as jnp

from netket.stats import statistics as mpi_statistics, mean as mpi_mean, Stats
from netket.utils.types import PyTree

from ._vjp import vjp as nkvjp


def expect(
    log_pdf: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    expected_fun: Callable[[PyTree, jnp.ndarray], jnp.ndarray],
    pars: PyTree,
    σ: jnp.ndarray,
    *expected_fun_args,
    n_chains: int = None,
) -> Tuple[jnp.ndarray, Stats]:
    """
    Computes the expectation value over a log-pdf.

    Args:
        log_pdf:
        expected_ffun
    """
    return _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args)


# log_prob_args and integrand_args are independent of params when taking the
# gradient. They can be continuous or discrete, and they can be pytrees
# Does not support higher-order derivatives yet
@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _expect(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ = L_σ.reshape((n_chains, -1))

    L̄_σ = mpi_statistics(L_σ.T)
    # L̄_σ = L_σ.mean(axis=0)

    return L̄_σ.mean, L̄_σ


def _expect_fwd(n_chains, log_pdf, expected_fun, pars, σ, *expected_fun_args):
    L_σ = expected_fun(pars, σ, *expected_fun_args)
    if n_chains is not None:
        L_σ_r = L_σ.reshape((n_chains, -1))
    else:
        L_σ_r = L_σ

    L̄_stat = mpi_statistics(L_σ_r.T)

    L̄_σ = L̄_stat.mean
    # L̄_σ = L_σ.mean(axis=0)

    # Use the baseline trick to reduce the variance
    ΔL_σ = L_σ - L̄_σ

    return (L̄_σ, L̄_stat), (pars, σ, expected_fun_args, ΔL_σ)


# TODO: in principle, the gradient of an expectation is another expectation,
# so it should support higher-order derivatives
# But I don't know how to transform log_prob_fun into grad(log_prob_fun) while
# keeping the chunk dimension and without a loop through the chunk dimension
def _expect_bwd(n_chains, log_pdf, expected_fun, residuals, dout):
    pars, σ, cost_args, ΔL_σ = residuals
    dL̄, dL̄_stats = dout

    def f(pars, σ, *cost_args):
        log_p = log_pdf(pars, σ)
        term1 = jax.vmap(jnp.multiply)(ΔL_σ, log_p)
        term2 = expected_fun(pars, σ, *cost_args)
        out = mpi_mean(term1 + term2, axis=0)
        out = out.sum()
        return out

    _, pb = nkvjp(f, pars, σ, *cost_args)
    grad_f = pb(dL̄)
    return grad_f


_expect.defvjp(_expect_fwd, _expect_bwd)
