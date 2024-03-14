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

import flax.linen as nn
import jax.numpy as jnp

from netket.hilbert import DiscreteHilbert
from netket.utils.types import DType, Array, NNInitFunc


class LogStateVector(nn.Module):
    r"""
    _Exact_ ansatz storing the logarithm of the full, exponentially large
    wavefunction coefficients. As with other models, coefficients do not need
    to be normalised.

    This ansatz can only be used with Hilbert spaces which are small enough to
    be indexable.

    By default it initialises as a uniform state.
    """

    hilbert: DiscreteHilbert
    """The Hilbert space."""

    param_dtype: DType = jnp.complex128
    """The dtype of the weights."""

    logstate_init: NNInitFunc = nn.initializers.ones
    """Initializer for the weights."""

    def setup(self):
        if not self.hilbert.is_indexable:
            raise ValueError(
                "StateVector can only be used with indexable hilbert spaces."
            )

        self.logstate = self.param(
            "logstate", self.logstate_init, (self.hilbert.n_states,), self.param_dtype
        )

    def __call__(self, x_in: Array):
        return self.logstate[self.hilbert.states_to_numbers(x_in)]
