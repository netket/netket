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
from typing import Tuple

from flax import linen as nn
from flax.linen.dtypes import promote_dtype

import jax
from jax import lax
from jax import numpy as jnp

from netket.nn.rnn import RNNLayer
from netket.utils import deprecate_dtype
from netket.utils.types import Array


# TODO: Is there a util function for this?
def _tree_where(cond, xs, ys):
    return jax.tree_util.tree_map(lambda x, y: jnp.where(cond, x, y), xs, ys)


def _get_h_xy_single(hiddens: Array, index: int, L: int) -> Tuple[Array, Array]:
    """
    Hard-coded snake ordering.

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
    return _tree_where(
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
          inputs: input data with dimensions (batch, length, features) and unordered layout.

        Returns:
          The output sequences.
        """
        batch_size, V, _ = inputs.shape
        L = int(sqrt(V))
        recur_func = self._get_recur_func(inputs)

        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        # We do not reorder the inputs before the scan, because we need to
        # access spatial neighbors

        def scan_func(carry, k):
            cell, outputs = carry
            index = self.reorder_idx[k]

            if self.exclusive:
                # Get the inputs at the previous site in the autoregressive order,
                # or zeros for the first site
                inputs_i = inputs[:, self.reorder_idx[k - 1], :]
                zeros = jnp.zeros_like(inputs_i)
                inputs_i = jnp.where(k == 0, zeros, inputs_i)
            else:
                inputs_i = inputs[:, index, :]

            h_x, h_y = _get_h_xy(outputs, index, L)
            cell, hidden = recur_func(inputs_i, cell, h_x, h_y)

            outputs = outputs.at[:, index, :].set(hidden)
            return (cell, outputs), outputs

        cell = jnp.zeros((batch_size, self.features), dtype=inputs.dtype)
        outputs = jnp.zeros((batch_size, V, self.features), dtype=inputs.dtype)
        (_, outputs), _ = lax.scan(scan_func, (cell, outputs), jnp.arange(V))
        return outputs


@deprecate_dtype
class LSTMLayer2D(RNNLayer2D):
    """2D long short-term memory layer."""

    def _get_recur_func(self, inputs):
        in_features = inputs.shape[-1]
        kernel, bias = self._dense_params(
            None, in_features + self.features * 2, self.features * 4
        )
        inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=None)

        def recur_func(inputs, cell, h_x, h_y):
            in_cat = jnp.concatenate([inputs, h_x, h_y], axis=-1)
            ifgo = nn.sigmoid(in_cat @ kernel + bias)
            i, f, g, o = ifgo.split(4, axis=-1)

            # sigmoid -> tanh
            g = g * 2 - 1

            cell = f * cell + i * g
            outputs = o * nn.tanh(cell)
            return cell, outputs

        return recur_func
