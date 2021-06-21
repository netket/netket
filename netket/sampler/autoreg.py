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

from netket.sampler import Sampler, SamplerState
from netket.utils import struct
from netket.utils.types import PRNGKeyT, PyTree

import jax
from jax import numpy as jnp


def batch_choice(key, a, p):
    """
    Batched version of `jax.random.choice`.

    Attributes:
      key: a PRNGKey used as the random key.
      a: 1D array. Random samples are generated from its elements.
      p: 2D array of shape `(batch_size, a.size)`. Each slice `p[i, :]` is
        the probabilities associated with entries in `a` to generate a sample
        at the index `i` of the output. Can be unnormalized.

    Returns:
      The generated samples as an 1D array of shape `(batch_size,)`.
    """
    p_cumsum = p.cumsum(axis=1)
    r = p_cumsum[:, -1:] * jax.random.uniform(key, shape=(p.shape[0], 1))
    indices = (r > p_cumsum).sum(axis=1)
    out = a[indices]
    return out


@struct.dataclass
class ARDirectSamplerState(SamplerState):
    σ: jnp.ndarray
    """current batch of (maybe partially sampled) configurations."""
    cache: PyTree
    """auxiliary states, e.g., used to implement fast autoregressive sampling."""
    key: PRNGKeyT
    """state of the random number generator."""

    def __repr__(self):
        return f"{type(self).__name__}(rng state={self.key})"


@struct.dataclass
class ARDirectSampler(Sampler):
    """Direct sampler for autoregressive neural networks."""

    @property
    def is_exact(sampler):
        return True

    def _init_cache(sampler, model, σ, key):
        variables = model.init(key, σ)
        if "cache" in variables:
            _, cache = variables.pop("cache")
        else:
            cache = None
        return cache

    def _init_state(sampler, model, params, key):
        new_key, key = jax.random.split(key)
        σ = jnp.zeros(
            (sampler.n_chains_per_rank, sampler.hilbert.size), dtype=sampler.dtype
        )
        cache = sampler._init_cache(model, σ, key)
        return ARDirectSamplerState(σ=σ, cache=cache, key=new_key)

    def _reset(sampler, model, params, state):
        return state

    def _sample_chain(sampler, model, params, state, chain_length):
        return _sample_chain(sampler, model, params, state, chain_length)

    def _sample_next(sampler, model, params, state):
        σ, new_state = sampler._sample_chain(model, params, state, 1)
        σ = σ.squeeze(axis=0)
        return new_state, σ


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(sampler, model, params, state, chain_length):
    def scan_fun(carry, index):
        σ, cache, key = carry
        new_key, key = jax.random.split(key)

        p, cache = model.apply(
            params,
            σ,
            cache,
            method=model.conditionals,
        )
        local_states = jnp.asarray(sampler.hilbert.local_states, dtype=sampler.dtype)
        p = p[:, index, :]
        new_σ = batch_choice(key, local_states, p)
        σ = σ.at[:, index].set(new_σ)

        return (σ, cache, new_key), None

    new_key, key_init, key_scan = jax.random.split(state.key, 3)

    # We just need a buffer for `σ` before generating each sample
    # The result does not depend on the initial contents in it
    σ = jnp.zeros(
        (chain_length * sampler.n_chains_per_rank, sampler.hilbert.size),
        dtype=sampler.dtype,
    )

    # Init `cache` before generating each sample,
    # even if `params` is not changed and `reset` is not called
    cache = sampler._init_cache(model, σ, key_init)

    indices = jnp.arange(sampler.hilbert.size)
    (σ, cache, _), _ = jax.lax.scan(
        scan_fun,
        (σ, cache, key_scan),
        indices,
    )
    σ = σ.reshape((chain_length, sampler.n_chains_per_rank, sampler.hilbert.size))

    new_state = state.replace(σ=σ, cache=cache, key=new_key)
    return σ, new_state
