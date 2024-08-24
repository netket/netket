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

"""
This module contains functions that act on hilbert spaces necessary
for sampling.

If you define custom hilbert spaces, and want to sample from it, you
should read carefully.

This module *exports* two functions:
 - `random_state(hilbert, key, batch_size, dtype)` which generates a batch of random
 states in the given hilbert space, in an array with the specified dtype.
 - `flip_state(hilb, key, states, indices)` which, for every state σ in the batch of
 states, considers σᵢ and returns a new state where that entry is different from the
 previous. The new configuration is selected with uniform probability among the local
 possible configurations.

Hilbert spaces must at least implement a `random_state` to support sampling.
`flip_state` is only necessary in order to use LocalRule samplers.
All hilbert spaces in netket implement both.

How to implement the two functions above
----------------------------------------

While the *exported* function acts on batches, sometimes it is hard to implement
the function on batches. Therefore they can be implemented either as a scalar
function, that gets jax.vmap-ed automatically, or directly as a batched rule.
Of course, if you implement the batched rule you will most likely see better
performance.

In order to implement the scalar rule for your custom hilbert object `MyHilbert`
you should define a function taking 4 inputs, the hilbert space, a jax PRNG key and
the dtype of the desired result. For random operations you should use the key
provided.

@netket.utils.dispatch.dispatch
def random_state(hilb: MyHilbert, key, dtype):
    return mystate

@netket.utils.dispatch.dispatch
def random_state(hilb: MyHilbert, key, batches: int, dtype):
    return mystate

flip_state is implemented in the same way, through

@netket.utils.dispatch.dispatch
def flip_state_scalar(hilb: Fock, key, σ, idx):
    return new_state, oldval

There is a vmapped default fallback for the batched version.

@netket.utils.dispatch.dispatch
def flip_state_batch(hilb: Fock, key, σ, idx):
    return new_states, oldvals

"""

from netket.utils import _hide_submodules

from . import custom, doubled, homogeneous, fock, qubit, tensor_hilbert, particle
from .base import flip_state, random_state

_hide_submodules(__name__)
