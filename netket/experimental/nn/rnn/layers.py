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
from jax import numpy as jnp

from netket.utils import HashableArray

from .cells import RNNCell
from .ordering import check_reorder_idx


class RNNLayer(nn.Module):
    """Recurrent neural network layer that maps inputs at N sites to outputs at N sites."""

    cell: RNNCell
    """cell to update the hidden memory at each site, such as LSTM or GRU."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    reorder_idx: HashableArray | None = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details.
    """
    inv_reorder_idx: HashableArray | None = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details.
    """
    prev_neighbors: HashableArray | None = None
    """previous neighbors of each site."""
    unroll: int = 1
    """how many steps to unroll in the recurrent loop. Trades compile time for
    faster runtime when networks are small."""

    def __post_init__(self):
        super().__post_init__()
        check_reorder_idx(self.reorder_idx, self.inv_reorder_idx, self.prev_neighbors)

    def _extract_inputs_i(self, inputs, k, index, prev_index):
        assert inputs.ndim == 3

        # Masking for 'exclusive' behaviour of first layer
        # TODO: Use 0 in masked sites or a value from computational basis?
        if self.exclusive:
            # Get the inputs at the previous site in the autoregressive order,
            # or zeros for the first site
            inputs_i = inputs[:, prev_index, :]
            inputs_i = jnp.where(k == 0, 0, inputs_i)
        else:
            # Get the inputs at the current site
            inputs_i = inputs[:, index, :]
        return inputs_i

    def _extract_hidden(self, outputs, index, prev_neighbors):
        assert outputs.ndim == 3

        if self.reorder_idx is None:
            # Get the hidden memory at the previous site,
            # or zeros for the first site
            hidden = outputs[:, index - 1, :]
            hidden = jnp.expand_dims(hidden, axis=-1)
        else:
            # Get the hidden memories at the previous neighbors
            prev_neighbors_i = prev_neighbors[index]
            hidden = outputs[:, prev_neighbors_i, :]
            # mask out inexistant previous neighbords
            hidden = jnp.where(prev_neighbors_i[None, :, None] == -1, 0, hidden)
        return hidden

    def __call__(self, inputs):
        """
        Applies the RNN cell to a batch of input sequences.

        Args:
          inputs: input data with dimensions (batch, n_sites, features).

        Returns:
          The output sequences.
        """
        if inputs.ndim != 3:
            raise ValueError(
                "Requires 3 dimensions of (batch, n_sites, features). "
                "If you have no features, set it to 1."
            )

        batch_size, N, _ = inputs.shape
        inputs = promote_dtype(inputs, dtype=self.cell.param_dtype)[0]

        if self.reorder_idx is None:
            reorder_idx = None
            prev_neighbors = None
        else:
            reorder_idx = jnp.asarray(self.reorder_idx)
            prev_neighbors = jnp.asarray(self.prev_neighbors)

        def scan_func(rnn_cell, carry, k):
            cell_mem, outputs = carry
            if self.reorder_idx is None:
                index = k
                prev_index = k - 1
            else:
                index = reorder_idx[k]
                prev_index = reorder_idx[k - 1]

            inputs_i = self._extract_inputs_i(inputs, k, index, prev_index)
            hidden = self._extract_hidden(outputs, index, prev_neighbors)
            cell_mem, hidden = rnn_cell(inputs_i, cell_mem, hidden)
            outputs = outputs.at[:, index, :].set(hidden)
            return (cell_mem, outputs), None

        scan = nn.scan(
            scan_func,
            variable_broadcast="params",
            split_rngs={"params": False},
            unroll=self.unroll,
        )

        cell_mem = jnp.zeros((batch_size, self.cell.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, N, self.cell.features), dtype=inputs.dtype)
        (_, outputs), _ = scan(self.cell, (cell_mem, outputs), jnp.arange(N))
        return outputs
