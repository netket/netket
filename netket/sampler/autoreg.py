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
    model_state: PyTree
    key: PRNGKeyT

    def __repr__(self):
        return f'ARSamplerState(rng state={self.key})'


@struct.dataclass
class ARSampler(Sampler):
    """Sampler for autoregressive neural networks."""
    def _init_state(sampler, model, params, key):
        return ARSamplerState(model_state=None, key=key)

    def _reset(sampler, model, params, state):
        return state.replace(model_state=None)

    def _sample_next(sampler, model, params, state):
        new_key, key = jax.random.split(state.key)

        def scan_fun(carry, index):
            spins, model_state, key = carry
            new_key, key = jax.random.split(key)

            p, model_state = model.apply(
                params,
                spins,
                model_state,
                method=model.conditionals,
            )
            local_states = jnp.asarray(sampler.hilbert.local_states,
                                       dtype=sampler.dtype)
            p = p[:, index, :]
            new_spins = vmap_choice(key, local_states, p)
            spins = spins.at[:, index].set(new_spins)

            return (spins, model_state, new_key), None

        spins, model_state = model.apply(
            params,
            (sampler.n_chains, sampler.hilbert.size),
            sampler.dtype,
            method=model.init_sample,
        )
        indices = jnp.arange(sampler.hilbert.size)
        (spins, model_state, _), _ = jax.lax.scan(
            scan_fun,
            (spins, model_state, key),
            indices,
        )

        new_state = state.replace(model_state=model_state, key=new_key)
        return new_state, spins
