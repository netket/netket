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

import jax
from jax import numpy as jnp


from functools import partial


# No longer implemented. See the generic implementation in
# homogeneous.py
#
# @dispatch
# @partial(jax.jit, static_argnames=("hilb", "batches", "dtype"))
# def random_state(  # noqa: F811
#    hilb: Fock, constraint: SumConstraint, key, batches: int, *, dtype=None
# ):
#    return _random_states_with_constraint_fock(
#        hilb.n_particles, hilb.shape, key, (batches,), dtype
#    )


def _choice(key, p):
    """
    Replacement for jax.random.choice with the following differences:

    - p can only contain 0 or 1
    - the return value is not a number i, but a mask with a single 1 at index i,
      and the rest 0
    - it supports arbitrary leading batch axes for p

    Args:
        key: a jax.random.PRNGKey
        p: an integer/boolean vector of probabilities, can only contain 0 or 1
           and does not have to sum up to 1. Can have arbitrary many leading batch dimensions
    Returns:
        A mask with a single 1, selecting one of the 1's in every vector of p uniformly

    It is functionally equivalent to, but more efficient than the following function:

    import jax
    import jax.numpy as jnp

    def _choice_jax(key, p):
        p_flat = p.reshape(-1, p.shape[-1])
        @jax.vmap
        def _f(k,p):
            return jax.random.choice(k, len(p), p=p) == jnp.arange(len(p))
        keys = jax.random.split(key, len(p_flat))
        return _f(keys, p_flat).reshape(p.shape)
    """
    # p needs to be in [0, 1], of type integer or bool
    # in the following all the sites are indexed starting from 1
    # to distinguish between site 0 (now 1) and not selecting it
    # e.g  take p = [[1 0 0 1 0 1 1 0]]
    cs = jnp.cumsum(p, axis=-1)  # now  cs = [[1 1 1 2 2 3 4 4]]
    n_candidates = cs[..., -1]  # == p.sum(axis=-1, keepdims=True)
    # 1 is exlusive in random.uniform
    # +1 because we index starting from 1
    r = jax.random.uniform(key, p.shape[:-1]) * n_candidates + 1
    # now cs*p = [[1 0 0 2 0 3 4 0]] and floor(r) in [1,2,3,4]
    return (cs * p) == jax.lax.floor(r).astype(cs.dtype)[..., None]


@partial(jax.jit, static_argnames=("n_particles", "hilb_shape", "shape", "dtype"))
def _random_states_with_constraint_fock(n_particles, hilb_shape, key, shape, dtype):
    # Distribute hilb.n_particles onto hilb.size sites
    # and put at most hilb.shape-1 particles in every site.
    # Note that this is NOT a uniform distribution over the
    # basis states of the constrained hilbert space.

    assert n_particles is not None
    hilb_size = len(hilb_shape)

    # start with all sites empty
    init = jnp.zeros(shape + (hilb_size,), dtype=dtype)

    # if constrained and uniformly n_max == 2, use a trick to sample quickly
    if set(hilb_shape) == {2}:
        return jax.random.permutation(
            key, init.at[..., :n_particles].set(1), axis=-1, independent=True
        )

    # shape is per site n_max
    n_max = jnp.array(hilb_shape) - 1

    def body_fun(x, key):
        # select all sites which are not yet full
        p = x < n_max
        # uniformly select a site among the not yet full ones
        # and put a particle in it
        carry = x + _choice(key, p)
        return carry, None

    # iterate body_fun above n_particles times
    keys = jax.random.split(key, n_particles)
    return jax.lax.scan(body_fun, init, keys)[0]
