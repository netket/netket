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

import abc

from flax import linen as nn
from flax.linen.dtypes import promote_dtype
from jax import numpy as jnp
from jax.nn.initializers import orthogonal, zeros

from netket.utils.types import DType, NNInitFunc


default_kernel_init = orthogonal()


class RNNCell(nn.Module):
    """Recurrent neural network cell that updates the hidden memory at each site."""

    features: int
    """output feature density, should be the last dimension."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""

    @abc.abstractmethod
    def __call__(self, inputs, cell_mem, hidden):
        """
        Applies the RNN cell to a batch of input sites at a given index.

        Args:
            inputs: input data with dimensions (batch, in_features).
            cell_mem: cell memory from the previous site with dimensions (batch, features).
            hidden: hidden memories from the previous neighbors with dimensions (batch, n_neighbors, features).

        Returns:
             * :code:`cell_mem`
                the updated cell memory with dimensions :code:`(batch, self.features)`.
             * :code:`outputs`
                the updated hidden memory with dimensions :code:`(batch, self.features)`,
                also serves as the output data at the current site for the
                :class:`netket.experimental.nn.rnn.RNNLayer` layer.
        """


class LSTMCell(RNNCell):
    """Long short-term memory cell."""

    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
        batch_size = inputs.shape[0]
        in_features = inputs.shape[-1]

        hidden = hidden.reshape((batch_size, -1))
        hid_features = hidden.shape[-1]
        in_cat = jnp.concatenate([inputs, hidden], axis=-1)

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

        cell_mem = f * cell_mem + i * g
        outputs = o * nn.tanh(cell_mem)
        return cell_mem, outputs


class GRU1DCell(RNNCell):
    """Gated recurrent unit cell.

    Only supports one previous neighbor at each site.
    """

    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weight matrix."""
    bias_init: NNInitFunc = zeros
    """initializer for the bias."""

    # cell_mem is not used
    @nn.compact
    def __call__(self, inputs, cell_mem, hidden):
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
            (in_features + hid_features, self.features),
            self.param_dtype,
        )
        n_bias = self.param(
            "n_bias",
            self.bias_init,
            (self.features),
            self.param_dtype,
        )

        in_cat, rz_kernel, rz_bias, n_kernel, n_bias = promote_dtype(
            in_cat, rz_kernel, rz_bias, n_kernel, n_bias, dtype=None
        )

        rz = nn.sigmoid(in_cat @ rz_kernel + rz_bias)
        r, z = jnp.split(rz, 2, axis=-1)

        in_cat = jnp.concatenate([inputs, r * hidden], axis=-1)
        n = nn.tanh(in_cat @ n_kernel + n_bias)

        outputs = (1 - z) * n + z * hidden
        return cell_mem, outputs
