from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp

from netket.utils.types import Array, PyTree
from netket.jax import apply_chunked
from netket.stats import mean as distributed_mean

from advanced_drivers._src.driver.ngd.driver_abstract_ngd import _flatten_samples


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "afun_t",
        "chunk_size_U",
        "chunk_size_V",
    ),
)
def smc_kernel(
    afun: Callable,
    vars: PyTree,
    samples: Array,
    afun_t: Callable,
    vars_t: PyTree,
    samples_t: Array,
    weights_t: Array,
    cv_coeff: Optional[float] = -0.5,
    chunk_size_U: Optional[int] = None,
    chunk_size_V: Optional[int] = None,
):
    afun_ = apply_chunked(partial(afun, vars), chunk_size=chunk_size_V)
    afun_t_ = apply_chunked(partial(afun_t, vars_t), chunk_size=chunk_size_U)

    out_shape = samples.shape[:-1]

    # equivalent to .reshape(-1, N)
    samples = _flatten_samples(samples)
    samples_t = _flatten_samples(samples_t)

    logVψ_x = afun_(samples)
    logUϕ_x = afun_t_(samples)
    logVψ_y = afun_(samples_t)
    logUϕ_y = afun_t_(samples_t)

    logRϕψ = logUϕ_x - logVψ_x
    logRψϕ = logVψ_y - logUϕ_y
    logA = logRϕψ + logRψϕ
    A = jnp.exp(logA)

    local_grad = (-1.0 * A).reshape(out_shape)

    if cv_coeff is not None:
        A = A.real + cv_coeff * (jnp.abs(A) ** 2 - 1)

    local_loss = (1 - A).reshape(out_shape)

    return local_grad * weights_t, local_loss * weights_t


@partial(
    jax.jit,
    static_argnames=(
        "afun",
        "afun_t",
        "chunk_size_U",
        "chunk_size_V",
    ),
)
def cmc_kernel(
    afun: Callable,
    vars: PyTree,
    samples: Array,
    afun_t: Callable,
    vars_t: PyTree,
    samples_t: Array,
    weights_t: Array,
    cv_coeff: Optional[float] = -0.5,
    chunk_size_U: Optional[int] = None,
    chunk_size_V: Optional[int] = None,
):
    afun_ = apply_chunked(partial(afun, vars), chunk_size=chunk_size_V)
    afun_t_ = apply_chunked(partial(afun_t, vars_t), chunk_size=chunk_size_U)

    out_shape = samples.shape[:-1]

    # equivalent to .reshape(-1, N)
    samples = jax.lax.collapse(samples, 0, samples.ndim - 1)
    samples_t = jax.lax.collapse(samples_t, 0, samples_t.ndim - 1)

    logVψ_x = afun_(samples)
    logUϕ_x = afun_t_(samples)
    logVψ_y = afun_(samples_t)
    logUϕ_y = afun_t_(samples_t)

    Rϕψ_x = jnp.exp(logUϕ_x - logVψ_x)
    Rψϕ_y = jnp.exp(logVψ_y - logUϕ_y)

    E = distributed_mean(Rψϕ_y * weights_t)
    E2 = distributed_mean(jnp.abs(Rψϕ_y) ** 2 * weights_t)

    local_grad = (-1.0 * Rϕψ_x * E).reshape(out_shape)

    local_loss = Rϕψ_x * E
    if cv_coeff is not None:
        local_loss = local_loss.real + cv_coeff * (jnp.abs(Rϕψ_x) ** 2 * E2 - 1)
    local_loss = (1 - local_loss).reshape(out_shape)

    return local_grad, local_loss
