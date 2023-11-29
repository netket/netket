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

import flax
import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.hilbert import DiscreteHilbert
from netket.sampler import Sampler, SamplerState
from netket.utils.deprecation import warn_deprecation
from netket.utils.types import PRNGKeyT, DType


class ARDirectSamplerState(SamplerState):
    key: PRNGKeyT
    """state of the random number generator."""

    def __init__(self, key):
        self.key = key
        super().__init__()


class ARDirectSampler(Sampler):
    r"""
    Direct sampler for autoregressive neural networks.

    This sampler only works with Flax models.
    This flax model must expose a specific method, `model.conditional`, which given
    a batch of samples and an index `i∈[0,self.hilbert.size]` must return the vector
    of partial probabilities at index `i` for the various (partial) samples provided.

    In short, if your model can be sampled according to a probability
    $ p(x) = p_1(x_1)p_2(x_2|x_1)\dots p_N(x_N|x_{N-1}\dots x_1) $ then
    `model.conditional(x, i)` should return $p_i(x)$.

    NetKet implements some autoregressive networks that can be used together with this
    sampler.
    """

    def __init__(
        self,
        hilbert: DiscreteHilbert,
        machine_pow: None = None,
        dtype: DType = float,
        *,
        n_chains=None,
        n_chains_per_rank=None,
    ):
        """
        Construct an autoregressive direct sampler.

        Args:
            hilbert: The Hilbert space to sample.
            dtype: The dtype of the states sampled (default = np.float64).

        Note:
            `ARDirectSampler.machine_pow` has no effect. Please set the model's `machine_pow` instead.
        """
        if n_chains is not None or n_chains_per_rank is not None:
            warn_deprecation(
                "Specifying `n_chains` or `n_chains_per_rank` when constructing exact samplers is deprecated."
            )

        if machine_pow is not None:
            raise ValueError(
                "ARDirectSampler.machine_pow should not be used. Modify the model `machine_pow` directly."
            )

        if hilbert.constrained:
            raise ValueError(
                "Only unconstrained Hilbert spaces can be sampled autoregressively with "
                "this sampler. To sample constrained spaces, you must write your own (do get in "
                "touch with us. We are interested!)"
            )

        return super().__init__(hilbert, machine_pow=2, dtype=dtype)

    @property
    def is_exact(sampler):
        """
        Returns `True` because the sampler is exact.

        The sampler is exact if all the samples are exactly distributed according to the
        chosen power of the variational state, and there is no correlation among them.
        """
        return True

    def _init_cache(sampler, model, σ, key):
        variables = model.init(key, σ, 0, method=model.conditional)
        cache = variables.get("cache")
        return cache

    def _init_state(sampler, model, variables, key):
        return ARDirectSamplerState(key=key)

    def _reset(sampler, model, variables, state):
        return state

    @partial(jax.jit, static_argnums=(1, 4))
    def _sample_chain(sampler, model, variables, state, chain_length):
        if "cache" in variables:
            variables, _ = flax.core.pop(variables, "cache")
        variables_no_cache = variables

        def scan_fun(carry, index):
            σ, cache, key = carry
            if cache:
                variables = {**variables_no_cache, "cache": cache}
            else:
                variables = variables_no_cache
            new_key, key = jax.random.split(key)

            p, mutables = model.apply(
                variables,
                σ,
                index,
                method=model.conditional,
                mutable=["cache"],
            )
            cache = mutables.get("cache")

            local_states = jnp.asarray(
                sampler.hilbert.local_states, dtype=sampler.dtype
            )
            new_σ = nkjax.batch_choice(key, local_states, p)
            σ = σ.at[:, index].set(new_σ)

            return (σ, cache, new_key), None

        new_key, key_init, key_scan = jax.random.split(state.key, 3)

        # Initialize a buffer for `σ` before generating a batch of samples
        # The result should not depend on its initial content
        σ = jnp.zeros(
            (sampler.n_chains_per_rank * chain_length, sampler.hilbert.size),
            dtype=sampler.dtype,
        )

        if config.netket_experimental_sharding:
            σ = jax.lax.with_sharding_constraint(
                σ, jax.sharding.PositionalSharding(jax.devices()).reshape(-1, 1)
            )

        # Initialize `cache` before generating a batch of samples,
        # even if `variables` is not changed and `reset` is not called
        cache = sampler._init_cache(model, σ, key_init)
        if cache:
            variables = {**variables_no_cache, "cache": cache}
        else:
            variables = variables_no_cache

        indices = jnp.arange(sampler.hilbert.size)
        indices = model.apply(variables, indices, method=model.reorder)
        (σ, _, _), _ = jax.lax.scan(scan_fun, (σ, cache, key_scan), indices)
        σ = σ.reshape((sampler.n_chains_per_rank, chain_length, sampler.hilbert.size))

        new_state = state.replace(key=new_key)
        return σ, new_state
