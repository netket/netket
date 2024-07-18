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

from collections.abc import Iterable

from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.graph import AbstractGraph
from netket.models.autoreg import ARNNSequential, _get_feature_list
from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc

from netket.experimental.nn.rnn import (
    GRU1DCell,
    LSTMCell,
    RNNLayer,
    default_kernel_init,
    ensure_prev_neighbors,
)


class RNN(ARNNSequential):
    """
    Base class for recurrent neural networks.

    If either one of `reorder_idx` and `inv_reorder_idx` is unspecified,
    it can be deduced from another. If both are unspecified, they can be
    determined from `graph`.

    If :attr:`~netket.experimental.models.RNN.prev_neighbors` is unspecified,
    it can be determined from :attr:`~netket.experimental.models.RNN.graph` and
    :attr:`~netket.experimental.models.RNN.reorder_idx`.

    If all of :attr:`~netket.experimental.models.RNN.reorder_idx`,
    :attr:`~netket.experimental.models.RNN.inv_reorder_idx`,
    :attr:`~netket.experimental.models.RNN.prev_neighbors`, and
    :attr:`~netket.experimental.models.RNN.graph`
    are unspecified, there is a faster code path for 1D RNN.
    """

    layers: int
    """number of layers."""
    features: Iterable[int] | int
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: HashableArray | None = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    inv_reorder_idx: HashableArray | None = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    prev_neighbors: HashableArray | None = None
    """previous neighbors of each site.
    An integer array of shape `(hilbert.size, max_prev_neighbors)`.
    When the actual number of previous neighbors of a site is less than `max_prev_neighbors`,
    use -1 to denote zero paddings instead of memory from a neighbor."""
    graph: AbstractGraph | None = None
    """graph of the physical system."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

    def __post_init__(self):
        reorder_idx, inv_reorder_idx, prev_neighbors = ensure_prev_neighbors(
            reorder_idx=self.reorder_idx,
            inv_reorder_idx=self.inv_reorder_idx,
            prev_neighbors=self.prev_neighbors,
            graph=self.graph,
        )

        self.reorder_idx = reorder_idx
        self.inv_reorder_idx = inv_reorder_idx
        self.prev_neighbors = prev_neighbors
        super().__post_init__()

    def reorder(self, inputs: Array, axis: int = 0) -> Array:
        if self.reorder_idx is None:
            return inputs
        else:
            idx = jnp.asarray(self.reorder_idx)
            return inputs.take(idx, axis)

    def inverse_reorder(self, inputs: Array, axis: int = 0) -> Array:
        if self.inv_reorder_idx is None:
            return inputs
        else:
            idx = jnp.asarray(self.inv_reorder_idx)
            return inputs.take(idx, axis)


class LSTMNet(RNN):
    """Long short-term memory network."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            RNNLayer(
                cell=LSTMCell(
                    features=features[i],
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
            for i in range(self.layers)
        ]


class GRUNet1D(RNN):
    """Gated recurrent unit network. Only supports one previous neighbor at each site."""

    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            RNNLayer(
                cell=GRU1DCell(
                    features=features[i],
                    param_dtype=self.param_dtype,
                    kernel_init=self.kernel_init,
                    bias_init=self.bias_init,
                ),
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
            )
            for i in range(self.layers)
        ]
