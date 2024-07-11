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

from netket.utils.types import Array

from .layers import RNNLayer


class FastRNNLayer(RNNLayer):
    """
    Recurrent neural network layer with fast sampling.

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
        inputs = promote_dtype(inputs, dtype=self.cell.param_dtype)[0]

        if self.reorder_idx is None:
            prev_neighbors = None
        else:
            prev_neighbors = jnp.asarray(self.prev_neighbors)

        _cell_mem = self.variable(
            "cache",
            "cell_mem",
            zeros,
            None,
            (batch_size, self.cell.features),
            inputs.dtype,
        )
        _outputs = self.variable(
            "cache",
            "outputs",
            zeros,
            None,
            (batch_size, self.size, self.cell.features),
            inputs.dtype,
        )
        cell_mem = _cell_mem.value
        outputs = _outputs.value

        hidden = self._extract_hidden(outputs, index, prev_neighbors)
        cell_mem, hidden = self.cell(inputs, cell_mem, hidden)

        if not self.is_initializing():
            _cell_mem.value = cell_mem
            _outputs.value = outputs.at[:, index, :].set(hidden)

        return hidden

    def __call__(self, inputs: Array) -> Array:
        return RNNLayer.__call__(self, inputs)
