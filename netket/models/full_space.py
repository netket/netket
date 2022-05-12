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

import jax
from jax.nn.initializers import normal

import jax.experimental.host_callback as hcb

from netket.hilbert import DiscreteHilbert
from netket.utils.types import DType, Array, NNInitFunc


def states_to_numbers(hilbert, σ):
    # calls back into python
    return hcb.call(
        hilbert.states_to_numbers,
        σ,
        result_shape=jax.ShapeDtypeStruct(σ.shape[:-1], jnp.int64),
    )


class LogStateVector(nn.Module):
    r"""
    Jastrow wave function :math:`\Psi(s) = \exp(\sum_{ij} s_i W_{ij} s_j)`.

    The W matrix is stored as a non-symmetric matrix, and symmetrized
    during computation by doing :code:`W = W + W.T` in the computation.
    """

    hilbert: DiscreteHilbert

    dtype: DType = jnp.complex128
    """The dtype of the weights."""

    logstate_init: NNInitFunc = normal()
    """Initializer for the weights."""

    def setup(self):
        if not self.hilbert.is_indexable:
            raise ValueError(
                "StateVector can only be used with indexable hilbert spaces."
            )

        self.logstate = self.param(
            "logstate", self.logstate_init, (self.hilbert.n_states, ), self.dtype
        )

    def __call__(self, x_in: Array):
        return self.logstate[states_to_numbers(self.hilbert, x_in)]
