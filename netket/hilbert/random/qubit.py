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

from netket.hilbert import Qubit

from .base import register_flip_state_impl, register_random_state_impl


def random_state_qubit_batch_impl(hilb: Qubit, key, batches, dtype):
    rs = jax.random.randint(key, shape=(batches, hilb.size), minval=0, maxval=2)
    return jnp.asarray(rs, dtype=dtype)


## flips
def flip_state_scalar_spin(hilb: Qubit, key, x, i):
    return jax.ops.index_update(x, i, -x[i] + 1), x[i]


register_random_state_impl(Qubit, batch=random_state_qubit_batch_impl)
register_flip_state_impl(Qubit, scalar=flip_state_scalar_spin)
