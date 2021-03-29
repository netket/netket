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
This module constains functions that act on hilbert spaces necessary
for sampling.

If you define custom hilbert spaces, and want to sample from it, you
should read carefully.

This module *exports* two functions: 
 - `random_state(hilbert, key, batch_size, dtype)` which generates a batch of random
 states in the given hilbert space, in an array with the specified dtype. 
 - `flip_state(hilb, key, states, indices)` which, for every state σ in the batch of states,
 considers σᵢ and returns a new state where that entry is different from the previous.
 The new configuration is selected with uniform probability among the local possible
 configurations.

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
the dtype of the derised result. For random operations you should use the key 
provided.

def random_state_myhilbert_scalar_impl(hilb: MyHilbert, key, dtype):    
    return mystate

The batched version takes an extra argument, that is the number of batches to generate, 
an int.

def random_state_myhilbert_batch_impl(hilb: MyHilbert, key, batches, dtype):    
    return mystate

Then register the implementation with the following function:
batch can be None or your implementation. 

nk.hilbert.random.register_random_state_impl(MyHilbert, scalar=random_state_myhilbert_scalar_impl, batch=None)

flip_state is implemented in the same way, through the function register_flip_state_impl

"""

from .base import (
    random_state,
    flip_state,
    register_flip_state_impl,
    register_random_state_impl,
)

from . import custom
from . import qubit
from . import spin
from . import fock
from . import doubled
from . import tensor_hilbert

from netket.utils import _hide_submodules

_hide_submodules(__name__)
