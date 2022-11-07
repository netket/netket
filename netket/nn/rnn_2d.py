# Copyright 2022 The NetKet Authors - All rights reserved.
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

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

from jax import lax
from jax import numpy as jnp

from netket.nn.rnn import RNNLayer, LSTMLayer1D
from netket.utils import deprecate_dtype


class RNNLayer2D(RNNLayer):
    """Base class for 2D recurrent neural network layers."""

    @nn.compact
    def __call__(self, inputs):
        """
        Applies the RNN cell to a batch of input sequences.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The output sequences.
        """
        batch_size, V, _ = inputs.shape

        if self.reorder_idx is None:
            max_prev_neighbors = 1
        else:
            reorder_idx = jnp.asarray(self.reorder_idx)
            prev_neighbors = jnp.asarray(self.prev_neighbors)
            max_prev_neighbors = prev_neighbors.shape[1]

        recur_func = self._get_recur_func(inputs, max_prev_neighbors * self.features)
        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        def scan_func(carry, k):
            cell, outputs = carry
            if self.reorder_idx is None:
                index = k
            else:
                index = reorder_idx[k]

            if self.exclusive:
                # Get the inputs at the previous site in the autoregressive order,
                # or zeros for the first site
                if self.reorder_idx is None:
                    inputs_i = inputs[:, k - 1, :]
                else:
                    inputs_i = inputs[:, reorder_idx[k - 1], :]
                inputs_i = jnp.where(k == 0, 0, inputs_i)
            else:
                # Get the inputs at the current site
                inputs_i = inputs[:, index, :]

            if self.reorder_idx is None:
                # Get the hidden memory at the previous site,
                # or zeros for the first site
                hidden = outputs[:, k - 1, :]
                hidden = jnp.expand_dims(hidden, axis=-1)
            else:
                # Get the hidden memories at the previous neighbors,
                # or zeros for boundaries
                n = prev_neighbors[index]
                hidden = outputs[:, n, :]
                hidden = jnp.where(n[None, :, None] == -1, 0, hidden)

            cell, hidden = recur_func(inputs_i, cell, hidden)

            outputs = outputs.at[:, index, :].set(hidden)
            return (cell, outputs), outputs

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, V, self.features), dtype=inputs.dtype)
        (_, outputs), _ = lax.scan(scan_func, (cell, outputs), jnp.arange(V))
        return outputs


@deprecate_dtype
class LSTMLayer2D(RNNLayer2D):
    """2D long short-term memory layer."""

    def _get_recur_func(self, inputs, hid_features):
        return LSTMLayer1D._get_recur_func(self, inputs, hid_features)
