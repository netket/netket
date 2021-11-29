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

import chex
import jax
from jax import numpy as jnp


def canonicalize_dtype(dtype):
    """Canonicalise a dtype, skip if None."""
    if dtype is not None:
        return jax.dtypes.canonicalize_dtype(dtype)
    return dtype


def cast_tree(tree, dtype):
    """Cast tree to given dtype, skip if None."""
    if dtype is not None:
        return jax.tree_map(lambda t: t.astype(dtype), tree)
    else:
        return tree


def safe_int32_increment(count: chex.Numeric) -> chex.Numeric:
    """Increments int32 counter by one.
    Normally `max_int + 1` would overflow to `min_int`. This functions ensures
    that when `max_int` is reached the counter stays at `max_int`.
    Args:
      count: a counter to be incremented.
    Returns:
      A counter incremented by 1, or max_int if the maximum precision is reached.
    """
    chex.assert_type(count, jnp.int32)
    max_int32_value = jnp.iinfo(jnp.int32).max
    one = jnp.array(1, dtype=jnp.int32)
    return jnp.where(count < max_int32_value, count + one, max_int32_value)
