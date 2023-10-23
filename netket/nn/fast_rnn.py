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
from jax.nn.initializers import zeros

from netket.nn.rnn import GRULayer1D, LSTMLayer, RNNLayer
from netket.utils.types import Array


class FastRNNLayer(RNNLayer):
    """
    Base class for recurrent neural network layers with fast sampling.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    size: int = None
    """number of sites."""

    @nn.compact
    def update_site(self, inputs: Array, index: int) -> Array:
        """
        Applies the RNN cell to a batch of input sites at a given index,
        and stores the updated memories in the cache.

        Args:
          inputs: an input site with dimensions (batch, features).
          index: the index of the output site. The index of the input site should be `index - self.exclusive`.

        Returns:
          The output site with dimensions (batch, features).
        """
        batch_size = inputs.shape[0]

        if self.reorder_idx is None:
            max_prev_neighbors = 1
        else:
            prev_neighbors = jnp.asarray(self.prev_neighbors)
            max_prev_neighbors = prev_neighbors.shape[1]

        recur_func = self._get_recur_func(inputs, max_prev_neighbors * self.features)
        inputs = promote_dtype(inputs, dtype=self.param_dtype)[0]

        _cell = self.variable(
            "cache", "cell", zeros, None, (batch_size, self.features), inputs.dtype
        )
        _outputs = self.variable(
            "cache",
            "outputs",
            zeros,
            None,
            (batch_size, self.size, self.features),
            inputs.dtype,
        )
        outputs = _outputs.value

        if self.reorder_idx is None:
            # Get the hidden memory at the previous site,
            # or zeros for the first site
            hidden = outputs[:, index - 1, :]
            hidden = jnp.expand_dims(hidden, axis=-1)
        else:
            # Get the hidden memories at the previous neighbors,
            # or zeros for boundaries
            n = prev_neighbors[index]
            hidden = outputs[:, n, :]
            hidden = jnp.where(n[None, :, None] == -1, 0, hidden)

        cell, hidden = recur_func(inputs, _cell.value, hidden)

        initializing = self.is_mutable_collection("params")
        if not initializing:
            _cell.value = cell
            _outputs.value = outputs.at[:, index, :].set(hidden)

        return hidden

    def __call__(self, inputs: Array) -> Array:
        return RNNLayer.__call__(self, inputs)


class FastLSTMLayer(FastRNNLayer):
    """
    Long short-term memory layer with fast sampling.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def _get_recur_func(self, inputs, hid_features):
        return LSTMLayer._get_recur_func(self, inputs, hid_features)


class FastGRULayer1D(FastRNNLayer):
    """
    Gated recurrent unit layer with fast sampling. Only supports one previous neighbor at each site.

    See :class:`netket.models.FastARNNSequential` for a brief explanation of fast autoregressive sampling.
    """

    def _get_recur_func(self, inputs, hid_features):
        return GRULayer1D._get_recur_func(self, inputs, hid_features)
