# Copyright 2021 The NetKet Authors - All Rights Reserved.
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


import jax
import jax.numpy as jnp

import netket.jax as nkjax
from netket.utils.types import Array, PyTree


LimitsDType = tuple[float | None, float | None]
"""Type of the dt limits field, having independently optional upper and lower bounds."""


def expand_dim(tree: PyTree, sz: int) -> PyTree:
    """
    creates a new pytree with same structure as input `tree`, but where very leaf
    has an extra dimension at 0 with size `sz`.
    """

    def _expand(x):
        return jnp.zeros((sz, *x.shape), dtype=x.dtype)

    return jax.tree_util.tree_map(_expand, tree)


def propose_time_step(
    dt: float, scaled_error: float, error_order: int, limits: LimitsDType
) -> float:
    """
    Propose an updated dt based on the scheme suggested in Numerical Recipes, 3rd ed.
    """
    SAFETY_FACTOR = 0.95
    err_exponent = -1.0 / (1 + error_order)
    return jnp.clip(
        dt * SAFETY_FACTOR * scaled_error**err_exponent,
        limits[0],
        limits[1],
    )


def set_flag_jax(condition, flags, flag):
    """
    If `condition` is true, `flags` is updated by setting `flag` to 1.
    This is equivalent to the following code, but compatible with jax.jit:
        if condition:
            flags |= flag
    """
    return jax.lax.cond(
        condition,
        lambda x: x | flag,
        lambda x: x,
        flags,
    )


def scaled_error(y, y_err, atol, rtol, *, last_norm_y=None, norm_fn) -> float:
    norm_y = norm_fn(y)
    scale = (atol + jnp.maximum(norm_y, last_norm_y) * rtol) / nkjax.tree_size(y_err)
    return norm_fn(y_err) / scale, norm_y


def euclidean_norm(x: PyTree | Array) -> float:
    """
    Computes the Euclidean L2 norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 2), x),
            )
        )


def maximum_norm(x: PyTree | Array) -> float:
    """
    Computes the maximum norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.max(jnp.abs(x))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                jnp.maximum,
                jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), x),
            )
        )
