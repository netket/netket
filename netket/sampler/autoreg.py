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

import jax
from jax import numpy as jnp

from netket.sampler import Sampler, SamplerState
from netket.utils import struct
from netket.utils.deprecation import warn_deprecation
from netket.utils.types import PRNGKeyT


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
    key: PRNGKeyT
    """state of the random number generator."""

    def __repr__(self):
        return f"{type(self).__name__}(rng state={self.key})"


@struct.dataclass
class ARDirectSampler(Sampler):
    r"""
    Direct sampler for autoregressive neural networks.

    This sampler only works with Flax models.
    This flax model must expose a specific method, `model._conditional`, which given
    a batch of samples and an index `i∈[0,self.hilbert.size]` must return the vector
    of partial probabilities at index `i` for the various (partial) samples provided.

    In short, if your model can be sampled according to a probability
    $ p(x) = p_1(x_1)p_2(x_2|x_1)\dots p_N(x_N|x_{N-1}\dots x_1) $ then
    `model._conditional(x, i)` should return $p_i(x)$.

    NetKet implements some autoregressive networks that can be used together with this
    sampler.
    """

    def __pre_init__(self, *args, **kwargs):
        """
        Construct an autoregressive direct sampler.

        Args:
            hilbert: The Hilbert space to sample.
            dtype: The dtype of the states sampled (default = np.float64).

        Note:
            `ARDirectSampler.machine_pow` has no effect. Please set the model's `machine_pow` instead.
        """
        if "n_chains" in kwargs or "n_chains_per_rank" in kwargs:
            warn_deprecation(
                "Specifying `n_chains` or `n_chains_per_rank` when constructing exact samplers is deprecated."
            )

        return super().__pre_init__(*args, **kwargs)

    def __post_init__(self):
        super().__post_init__()

        # self.machine_pow may be traced in jit
        if isinstance(self.machine_pow, int) and self.machine_pow != 2:
            raise ValueError(
                "ARDirectSampler.machine_pow should not be used. Modify the model `machine_pow` directly."
            )

    @property
    def is_exact(sampler):
        """
        Returns `True` because the sampler is exact.

        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True

    def _init_cache(sampler, model, σ, key):
        variables = model.init(key, σ, 0, method=model._conditional)
        if "cache" in variables:
            cache = variables["cache"]
        else:
            cache = None
        return cache

    def _init_state(sampler, model, variables, key):
        return ARDirectSamplerState(key=key)

    def _reset(sampler, model, variables, state):
        return state

    def _sample_chain(sampler, model, variables, state, chain_length):
        return _sample_chain(sampler, model, variables, state, chain_length)


@partial(jax.jit, static_argnums=(1, 4))
def _sample_chain(sampler, model, variables, state, chain_length):
    """
    Internal method used for jitting calls.
    """
    if "cache" in variables:
        variables, _ = variables.pop("cache")

    def scan_fun(carry, index):
        σ, cache, key = carry
        if cache:
            _variables = {**variables, "cache": cache}
        else:
            _variables = variables
        new_key, key = jax.random.split(key)

        p, mutables = model.apply(
            _variables,
            σ,
            index,
            method=model._conditional,
            mutable=["cache"],
        )
        if "cache" in mutables:
            cache = mutables["cache"]
        else:
            cache = None

        local_states = jnp.asarray(sampler.hilbert.local_states, dtype=sampler.dtype)
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

    # Initialize `cache` before generating each sample,
    # even if `variables` is not changed and `reset` is not called
    cache = sampler._init_cache(model, σ, key_init)

    indices = jnp.arange(sampler.hilbert.size)
    (σ, _, _), _ = jax.lax.scan(scan_fun, (σ, cache, key_scan), indices)
    σ = σ.reshape((chain_length, sampler.n_chains_per_rank, sampler.hilbert.size))

    new_state = state.replace(key=new_key)
    return σ, new_state
