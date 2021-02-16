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

"""
This module constains functions that act on hilbert spaces necessary
for sampling.

This module exports two functions: 
 - `random_state(hilbert, key, batch_size, dtype)` which generates a batch of random
 states in the given hilbert space, in an array with the specified dtype. 
 - `flip_state(hilb, key, states, indices)` which, for every state σ in the batch of states,
 considers σᵢ and returns a new state where that entry is different from the previous.
 The new configuration is selected with uniform probability among the local possible
 configurations.

Those methods can be implemented both as scalar functions by overloading
`xxx_scalar_impl` or as batched functions by overloading `xxx_batch_impl`.
If only the scalar is defined, then jax.vmap is used to map over the batch
axis. Of course, it is usually more performant to define the batched code, 
but that is not always possible.

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
