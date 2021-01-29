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
from flax import struct

from netket.hilbert.random import flip_state

from ..metropolis import MetropolisRule


@struct.dataclass
class LocalRule(MetropolisRule):
    def transition(rule, sampler, machine, parameters, state, key, σ):
        key1, key2 = jax.random.split(key, 2)

        n_chains = σ.shape[0]
        hilb = sampler.hilbert
        local_states = jnp.array(np.sort(np.array(hilb.local_states)))

        indxs = jax.random.randint(key1, shape=(n_chains,), minval=0, maxval=hilb.size)
        σp, _ = flip_state(hilb, key2, σ, indxs)

        return σp, None
