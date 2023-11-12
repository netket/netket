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
from jax.nn.initializers import orthogonal, zeros

from netket.utils import HashableArray
from netket.utils.types import DType, NNInitFunc

from .ordering import check_reorder_idx

default_kernel_init = orthogonal()


class RNNLayer(nn.Module):
    """Base class for recurrent neural network layers."""

    cell: int
    """Recurrent cell, such as LSTM or GRU."""
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

    def setup(self):
        self.features = self.cell.features

        if self.prev_neighbors is not None:
            self.max_prev_neighbors = self.prev_neighbors.shape[1]
        else:
            self.max_prev_neighbors = 1

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
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The output sequences.
        """
        if inputs.ndim != 3:
            raise ValueError(
                "Requires 3 dimensions where (batch, Nsites, features)."
                "If you have no features, set it to 1."
            )

        batch_size, N, _ = inputs.shape
        inputs = self.reorder(inputs)

        def scan_func(rnn_unit, carry, k):
            cell, outputs = carry

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

            cell, hidden = rnn_unit(inputs_i, cell, hidden)

            outputs = outputs.at[:, k, :].set(hidden)
            carry = (cell, outputs)

            return carry, (outputs,)

        time_axis = 0
        scan = nn.scan(
            scan_func,
            in_axes=(0,),
            out_axes=time_axis,
            unroll=self.unroll,  # self.unroll,
            variable_axes={},
            variable_broadcast="params",
            variable_carry=False,
            split_rngs={"params": False},
        )

        cell, outputs = self.cell.initialize_carry(inputs)
        carry = (cell, outputs)

        inputsN = jnp.arange(N)

        scan_output = scan(self.cell, carry, inputsN)
        (_, outputs), _ = scan_output
        inputs = self.inverse_reorder(outputs)

        return outputs


class LSTMCell(nn.Module):
    """Long short-term memory layer."""

    features: int
    """output feature density, should be the last dimension."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def initialize_carry(self, inputs):
        batch_size, N, _ = inputs.shape

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, N, self.features), dtype=inputs.dtype)

        return (cell, outputs)

    @nn.compact
    def __call__(self, inputs, cell, hidden):
        batch_size = inputs.shape[0]
        in_features = inputs.shape[-1]

        hidden = hidden.reshape((batch_size, -1))
        in_cat = jnp.concatenate([inputs, hidden], axis=-1)

        hid_features = hidden.shape[-1]

        kernel = self.param(
            "kernel",
            self.kernel_init,
            (in_features + hid_features, self.features * 4),
            self.param_dtype,
        )
        bias = self.param(
            "bias",
            self.bias_init,
            (self.features * 4,),
            self.param_dtype,
        )

        in_cat, kernel, bias = promote_dtype(in_cat, kernel, bias, dtype=None)

        ifgo = nn.sigmoid(in_cat @ kernel + bias)
        i, f, g, o = jnp.split(ifgo, 4, axis=-1)

        # sigmoid -> tanh
        g = g * 2 - 1

        cell = f * cell + i * g
        outputs = o * nn.tanh(cell)
        return cell, outputs


class GRU1DCell(nn.Module):
    """Gated recurrent unit layer.

    Only supports one previous neighbor at each site.
    """

    features: int
    """output feature density, should be the last dimension."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    def initialize_carry(self, inputs):
        batch_size, N, _ = inputs.shape

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, N, self.features), dtype=inputs.dtype)

        cell, outputs, _ = jnp.promote_dtype(cell, outputs, inputs)

        return (cell, outputs)

    @nn.compact
    def __call__(self, inputs, cell, hidden):
        batch_size = inputs.shape[0]
        in_features = inputs.shape[-1]

        hidden = hidden.reshape((batch_size, -1))
        hid_features = hidden.shape[-1]

        in_cat = jnp.concatenate([inputs, hidden], axis=-1)

        rz_kernel = self.param(
            "rz_kernel",
            self.kernel_init,
            (in_features + hid_features, self.features * 2),
            self.param_dtype,
        )
        rz_bias = self.param(
            "rz_bias",
            self.bias_init,
            (self.features * 2,),
            self.param_dtype,
        )
        n_kernel = self.param(
            "n_kernel",
            self.kernel_init,
            (in_features + hid_features, self.features * 2),
            self.param_dtype,
        )
        n_bias = self.param(
            "n_bias",
            self.bias_init,
            (self.features * 2,),
            self.param_dtype,
        )

        in_cat, rz_kernel, rz_bias, n_kernel, n_bias = promote_dtype(
            in_cat, self.rz_kernel, self.rz_bias, self.n_kernel, self.n_bias, dtype=None
        )

        rz = nn.sigmoid(in_cat @ rz_kernel + rz_bias)
        r, z = jnp.split(rz, 2, axis=-1)

        in_cat = jnp.concatenate([inputs, r * hidden], axis=-1)
        n = nn.tanh(in_cat @ n_kernel + n_bias)

        outputs = (1 - z) * n + z * hidden
        return cell, outputs
