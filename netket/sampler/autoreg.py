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

from flax import struct
from netket.sampler import Sampler, SamplerState
from netket.utils.types import PRNGKeyT, PyTree

import jax
from jax import numpy as jnp


def vmap_choice(key, a, p, replace=True):
    """
    p.shape: (batch, a.shape)
    Return shape: (batch, )
    """

    def scan_fun(key, p_i):
        new_key, key = jax.random.split(key)
        out_i = jax.random.choice(key, a, replace=replace, p=p_i)
        return new_key, out_i

    _, out = jax.lax.scan(scan_fun, key, p)
    return out


@struct.dataclass
class ARSamplerState(SamplerState):
    σ: jnp.ndarray
    """current batch of (maybe partially sampled) configurations."""
    model_state: PyTree
    """auxiliary model state, e.g., used to implement fast autoregressive sampling."""
    key: PRNGKeyT
    """state of the random number generator."""

    def __repr__(self):
        return f"{type(self).__name__}(rng state={self.key})"


@struct.dataclass
class ARSampler(Sampler):
    """Sampler for autoregressive neural networks."""

    def _init_state(sampler, model, params, key):
        σ = sampler.hilbert.random_state(
            key, size=sampler.n_batches, dtype=sampler.dtype
        )
        model_state = model.apply(
            params,
            σ,
            method=model.init_state,
        )
        return ARSamplerState(σ=σ, model_state=model_state, key=key)

    def _reset(sampler, model, params, state):
        new_key, key = jax.random.split(state.key)
        σ = sampler.hilbert.random_state(
            key, size=sampler.n_batches, dtype=sampler.dtype
        )
        model_state = model.apply(
            params,
            σ,
            method=model.init_state,
        )
        return state.replace(σ=σ, model_state=model_state, key=new_key)

    def _sample_next(sampler, model, params, state):
        σ = state.σ
        model_state = state.model_state
        new_key, key = jax.random.split(state.key)

        def scan_fun(carry, index):
            σ, model_state, key = carry
            new_key, key = jax.random.split(key)

            p, model_state = model.apply(
                params,
                σ,
                model_state,
                method=model.conditionals,
            )
            local_states = jnp.asarray(
                sampler.hilbert.local_states, dtype=sampler.dtype
            )
            p = p[:, index, :]
            new_σ = vmap_choice(key, local_states, p)
            σ = σ.at[:, index].set(new_σ)

            return (σ, model_state, new_key), None

        indices = jnp.arange(sampler.hilbert.size)
        (σ, model_state, _), _ = jax.lax.scan(
            scan_fun,
            (σ, model_state, key),
            indices,
        )

        new_state = state.replace(σ=σ, model_state=model_state, key=new_key)
        return new_state, σ
