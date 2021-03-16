# Copyright 2021 The NetKet Authors - All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import flax
import numpy as np

from jax import numpy as jnp
from jax.experimental import host_callback as hcb
from flax import struct
from numba import jit

import math

from typing import Any

from netket.operator import AbstractOperator

from ..metropolis import MetropolisRule


@struct.dataclass
class HamiltonianRule(MetropolisRule):
    """
    Rule proposing moves according to the terms in an operator.

    In this case, the transition matrix is taken to be:

    .. math::
       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),
    """

    Ô: Any = struct.field(pytree_node=False)

    def __post_init__(self):
        # Raise errors if hilbert is not an Hilbert
        if not isinstance(self.Ô, AbstractOperator):
            raise TypeError(
                "Argument to HamiltonianRule must be a valid operator.".format(
                    type(self.Ô)
                )
            )

    def transition(rule, sampler, machine, parameters, state, key, σ):
        def _transition(args):
            # unpack arguments
            σ, rand_vec = args
            # rgen = np.random.default_rng(key)

            sections = np.empty(σ.shape[0], dtype=np.int32)
            σp, _ = rule.Ô.get_conn_flattened(σ, sections)

            σ_proposed, log_prob_corr = _choose(σp, sections, rand_vec)

            return σ_proposed, log_prob_corr - np.log(sections)

        # ideally we would pass the key to python/numba in _choose, initialise a
        # np.random.default_rng(key) and use it to generatee random uniform integers.
        # However, numba dose not support np states, and reseeding it's MT1998 implementation
        # would be slow so we generate floats in the [0,1] range in jax and pass those
        # to python
        rand_vec = jax.random.uniform(key, shape=(σ.shape[0],))

        σp, log_prob_correction = hcb.call(
            _transition,
            (σ, rand_vec),
            result_shape=(
                jax.ShapeDtypeStruct(σ.shape, σ.dtype),
                jax.ShapeDtypeStruct((σ.shape[0],), σ.dtype),
            ),
        )

        return σp, log_prob_correction

    def __repr__(self):
        return f"HamiltonianRule({self.Ô})"


@jit(nopython=True)
def _choose(σp, sections, rand_vec):
    out_σ = np.empty((len(sections), σp.shape[-1]), dtype=σp.dtype)
    out_log_prob_corr = np.zeros(sections.shape)

    low_range = 0
    for i, s in enumerate(sections):
        n_rand = int(rand_vec[i] * s)
        out_σ[i] = σp[n_rand]
        out_log_prob_corr[i] = math.log(s - low_range)
        low_range = s

    return out_σ, out_log_prob_corr
