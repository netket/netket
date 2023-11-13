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

from typing import Optional

from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from jax import numpy as jnp
from jax.nn.initializers import orthogonal

from netket.utils import HashableArray

from .cells import RNNCell
from .ordering import check_reorder_idx

default_kernel_init = orthogonal()


class RNNLayer(nn.Module):
    """Recurrent neural network layer that maps inputs at N sites to outputs at N sites."""

    cell: RNNCell
    """Cell to update the hidden memory at each site, such as LSTM or GRU."""
    exclusive: bool
    """True if an output element does not depend on the input element at the same index."""
    reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details.
    """
    inv_reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details.
    """
    prev_neighbors: Optional[HashableArray] = None
    """previous neighbors of each site."""
    unroll: int = 1
    """How much to unroll the recurrent loop. Trades compile time for
    faster runtime when networks are small."""

    def __post_init__(self):
        super().__post_init__()
        check_reorder_idx(self.reorder_idx, self.inv_reorder_idx, self.prev_neighbors)

    def reorder(self, states):
        assert states.ndim == 3
        if self.reorder_idx is not None:
            states = states[:, self.reorder_idx.wrapped, :]
        return states

    def inverse_reorder(self, states):
        assert states.ndim == 3
        if self.inv_reorder_idx is not None:
            states = states[:, self.inv_reorder_idx.wrapped, :]
        return states

    def extract_memory_state(self, outputs, k):
        assert outputs.ndim == 3
        assert k.ndim == 0
        if self.reorder_idx is None:
            # Get the hidden memory at the previous site,
            # or zeros for the first site
            hidden = outputs[:, k - 1, :]
            hidden = jnp.expand_dims(hidden, axis=-1)
        else:
            # convert previous neighbors to reordered previous neighbors
            prev_neighbors = jnp.asarray(
                self.inv_reorder_idx.wrapped[self.prev_neighbors.wrapped]
            )

            # Get the hidden memories at the previous neighbors
            prev_neighbors_k = prev_neighbors[k]
            hidden = outputs[:, prev_neighbors_k, :]
            # mask out inexistant previous neighbords
            hidden = jnp.where(prev_neighbors_k[None, :, None] == -1, 0, hidden)
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
        inputs = self.reorder(inputs)
        inputs = promote_dtype(inputs, dtype=self.cell.param_dtype)[0]

        def scan_func(rnn_cell, carry, k):
            cell_mem, outputs = carry

            # masking for 'exclusive' behaviour of first layer
            # TODO: Use 0 in masked sites or a value from computational basis?
            if self.exclusive:
                # Get the inputs at the previous site in the autoregressive order,
                # or zeros for the first site
                inputs_i = inputs[:, k - 1, :]
                inputs_i = jnp.where(k == 0, 0, inputs_i)
            else:
                # Get the inputs at the current site
                inputs_i = inputs[:, k, :]

            hidden = self.extract_memory_state(outputs, k)

            cell_mem, hidden = rnn_cell(inputs_i, cell_mem, hidden)
            outputs = outputs.at[:, k, :].set(hidden)
            return (cell_mem, outputs), (outputs,)

        scan = nn.scan(
            scan_func,
            unroll=self.unroll,
            variable_broadcast="params",
            split_rngs={"params": False},
        )

        cell_mem = jnp.zeros((batch_size, self.cell.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, N, self.cell.features), dtype=inputs.dtype)

        (_, outputs), _ = scan(self.cell, (cell_mem, outputs), jnp.arange(N))
        outputs = self.inverse_reorder(outputs)
        return outputs
