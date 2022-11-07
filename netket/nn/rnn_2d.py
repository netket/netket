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

from math import sqrt

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

import jax
from jax import lax
from jax import numpy as jnp

from netket.nn.rnn import RNNLayer, LSTMLayer1D
from netket.utils import deprecate_dtype
from netket.utils.types import Array


# TODO: Is there a util function for this?
def _tree_where(cond, xs, ys):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), xs, ys)


def _get_h_xy_single(hiddens: Array, index: int, L: int) -> Array:
    """
    Hard-coded snake ordering for square lattice.

    Given the index (i, j) of the current site, returns the hidden memories at
    the two previous neighboring sites (i, j ± 1) and (i - 1, j), or zeros if
    at the boundary.

    Args:
      hiddens: hidden memories with dimensions (length, features), unordered and
        can be partially filled.
    """

    def h(i, j):
        return hiddens[i * L + j]

    i, j = divmod(index, L)
    zeros = jnp.zeros_like(h(0, 0))
    h_x, h_y = _tree_where(
        i % 2 == 0,
        _tree_where(
            j == 0,
            _tree_where(
                i == 0,
                (zeros, zeros),
                (zeros, h(i - 1, j)),
            ),
            (h(i, j - 1), h(i - 1, j)),
        ),
        _tree_where(
            j == L - 1,
            (zeros, h(i - 1, j)),
            (h(i, j + 1), h(i - 1, j)),
        ),
    )
    hidden = jnp.concatenate([h_x, h_y], axis=-1)
    return hidden


_get_h_xy = jax.vmap(_get_h_xy_single, in_axes=(0, None, None))


class RNNLayer2D(RNNLayer):
    """Base class for 2D recurrent neural network layers."""

    @nn.compact
    def __call__(self, inputs):
        """
        Applies the RNN cell to a batch of input sequences.
        See :class:`netket.models.AbstractARNN` for the explanation of
        autoregressive ordering.

        Args:
          inputs: input data with dimensions (batch, length, features).

        Returns:
          The output sequences.
        """
        batch_size, V, _ = inputs.shape
        L = int(sqrt(V))
        recur_func = self._get_recur_func(inputs, self.features * 2)

        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        # We do not reorder the inputs before the scan, because we need to
        # access spatial neighbors
        arange = jnp.arange(V)
        if self.reorder_idx is None:
            indices = arange
        else:
            indices = jnp.asarray(self.reorder_idx)

        def scan_func(carry, k):
            cell, outputs = carry
            index = indices[k]

            if self.exclusive:
                # Get the inputs at the previous site in the autoregressive order,
                # or zeros for the first site
                inputs_i = inputs[:, indices[k - 1], :]
                zeros = jnp.zeros_like(inputs_i)
                inputs_i = jnp.where(k == 0, zeros, inputs_i)
            else:
                inputs_i = inputs[:, index, :]

            hidden = _get_h_xy(outputs, index, L)
            cell, hidden = recur_func(inputs_i, cell, hidden)

            outputs = outputs.at[:, index, :].set(hidden)
            return (cell, outputs), outputs

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, V, self.features), dtype=inputs.dtype)
        (_, outputs), _ = lax.scan(scan_func, (cell, outputs), arange)
        return outputs


@deprecate_dtype
class LSTMLayer2D(RNNLayer2D):
    """2D long short-term memory layer."""

    def _get_recur_func(self, inputs, hid_features):
        return LSTMLayer1D._get_recur_func(self, inputs, hid_features)
