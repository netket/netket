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
from typing import Optional, Union
from collections.abc import Iterable

import numpy as np
from jax import numpy as jnp
from jax.nn.initializers import zeros

from netket.graph import AbstractGraph, Lattice
from netket.models.autoreg import ARNNSequential, _get_feature_list
from netket.nn.rnn import GRULayer1D, LSTMLayer, default_kernel_init
from netket.utils import HashableArray
from netket.utils.types import Array, DType, NNInitFunc


class RNN(ARNNSequential):
    """
    Base class for recurrent neural networks.

    If either one of `reorder_idx` and `inv_reorder_idx` is unspecified,
    it can be deduced from another. If both are unspecified, they can be
    determined from `graph`.

    If `prev_neighbors` is unspecified, it can be determined from `graph` and
    `reorder_idx`.

    If all of `reorder_idx`, `inv_reorder_idx`, `prev_neighbors`, and `graph`
    are unspecified, there is a faster code path for 1D RNN.
    """

    layers: int
    """number of layers."""
    features: Union[Iterable[int], int]
    """output feature density in each layer. If a single number is given,
    all layers except the last one will have the same number of features."""
    reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from unordered to ordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    inv_reorder_idx: Optional[HashableArray] = None
    """indices to transform the inputs from ordered to unordered.
    See :meth:`netket.models.AbstractARNN.reorder` for details."""
    prev_neighbors: Optional[HashableArray] = None
    """previous neighbors of each site.
    An integer array of shape `(hilbert.size, max_prev_neighbors)`.
    When the actual number of previous neighbors of a site is less than `max_prev_neighbors`,
    use -1 to denote zero paddings instead of memory from a neighbor."""
    graph: Optional[AbstractGraph] = None
    """graph of the physical system."""
    param_dtype: DType = jnp.float64
    """the dtype of the computation (default: float64)."""
    kernel_init: NNInitFunc = default_kernel_init
    """initializer for the weights."""
    bias_init: NNInitFunc = zeros
    """initializer for the biases."""
    machine_pow: int = 2
    """exponent to normalize the outputs of `__call__`."""

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


class _LSTMNet(RNN):
    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            LSTMLayer(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


class _GRUNet1D(RNN):
    def setup(self):
        features = _get_feature_list(self)
        self._layers = [
            GRULayer1D(
                features=features[i],
                exclusive=(i == 0),
                reorder_idx=self.reorder_idx,
                inv_reorder_idx=self.inv_reorder_idx,
                prev_neighbors=self.prev_neighbors,
                param_dtype=self.param_dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for i in range(self.layers)
        ]


def _get_inv_idx(idx: Array) -> HashableArray:
    idx = np.asarray(idx)
    inv = np.empty_like(idx)
    for i, k in enumerate(idx):
        inv[k] = i
    inv = HashableArray(inv)
    return inv


def _get_extent(graph: AbstractGraph) -> tuple[int, int, int]:
    V = graph.n_nodes

    if isinstance(graph, Lattice):
        assert len(graph.extent) == 2
        L, M = graph.extent
    else:
        L = M = int(sqrt(V))
    assert L * M == V

    return V, L, M


def _get_snake_inv_reorder_idx(graph: AbstractGraph) -> HashableArray:
    V, L, M = _get_extent(graph)
    idx = np.arange(V, dtype=np.intp).reshape((L, M))
    idx[1::2, :] = idx[1::2, ::-1]
    idx = idx.flatten()
    idx = HashableArray(idx)
    return idx


def _get_inv_reorder_idx(graph: AbstractGraph) -> HashableArray:
    """
    A greedy algorithm to determine an autoregressive order with good locality.
    For any rectangular graph with OBC, it is the same as the usual snake ordering.
    For PBC, it is mostly the same except at the end of each row.

    Start from site 0;
    at each step, choose the unvisited neighbor whose index is the closest to the last one;
    if two neighbors have the same index distance, choose the one with the smaller index;
    if there is no unvisited neighbor, choose the unvisited site with the smallest index.

    Args:
      graph: A :class:`netket.graph.AbstractGraph` instance.

    Returns:
      A hashable array of ints describing the indices to transform an array from ordered to unordered.
      See :meth:`netket.models.AbstractARNN.reorder` for details.
    """
    V = graph.n_nodes
    adj = graph.adjacency_list()
    idx = np.empty(V, dtype=np.intp)
    visited = np.zeros(V, dtype=bool)

    idx[0] = 0
    visited[0] = True
    for i in range(1, V):
        last_k = idx[i - 1]
        neighbors = [x for x in adj[last_k] if not visited[x]]
        if neighbors:
            k = min([(abs(x - last_k), x) for x in neighbors])[1]
        else:
            k = next(x for x in range(V) if not visited[x])
        idx[i] = k
        visited[k] = True

    idx = HashableArray(idx)
    return idx


def _get_snake_prev_neighbors(graph: AbstractGraph) -> HashableArray:
    V, L, M = _get_extent(graph)

    def h(i, j):
        if 0 <= i < L and 0 <= j < M:
            return i * M + j
        else:
            return -1

    def get_neighbors(k):
        i, j = divmod(k, M)
        return h(i, j - 1 if i % 2 == 0 else j + 1), h(i - 1, j)

    # Sort and put -1 padding at the end
    n = [sorted(get_neighbors(k), key=lambda x: (x < 0, x)) for k in range(V)]
    n = np.asarray(n, dtype=np.intp)
    n = HashableArray(n)
    return n


def _get_prev_neighbors(
    graph: AbstractGraph, reorder_idx: Array, max_prev_neighbors=None
) -> HashableArray:
    adj = graph.adjacency_list()
    reorder_idx = np.asarray(reorder_idx)

    n = [[y for y in x if reorder_idx[y] < reorder_idx[i]] for i, x in enumerate(adj)]
    # By default we take the median number of previous neighbors, rounded up
    if max_prev_neighbors is None:
        max_prev_neighbors = int(np.median([len(x) for x in n]) + 0.5)
    # When a site has more previous neighbors, select the ones with largest indices
    n = [x[-max_prev_neighbors:] for x in n]
    n = [sorted(x) + [-1] * (max_prev_neighbors - len(x)) for x in n]

    n = np.asarray(n, dtype=np.intp)
    n = HashableArray(n)
    return n


def _ensure_prev_neighbors(kwargs):
    reorder_idx = kwargs.get("reorder_idx")
    inv_reorder_idx = kwargs.get("inv_reorder_idx")
    prev_neighbors = kwargs.get("prev_neighbors")
    graph = kwargs.get("graph")

    if inv_reorder_idx is None and graph is not None:
        inv_reorder_idx = _get_inv_reorder_idx(graph)

    if reorder_idx is None:
        if inv_reorder_idx is not None:
            reorder_idx = _get_inv_idx(inv_reorder_idx)
    else:
        if inv_reorder_idx is None:
            inv_reorder_idx = _get_inv_idx(reorder_idx)
    # Now reorder_idx and inv_reorder_idx must be both None or not None

    if reorder_idx is None:
        if prev_neighbors is None:
            # There is a faster code path for 1D RNN
            pass
        else:
            raise ValueError(
                "When `prev_neighbors` is provided, you must also provide "
                "either `reorder_idx` or `inv_reorder_idx`."
            )
    else:
        if prev_neighbors is None:
            if graph is None:
                raise ValueError(
                    "When `reorder_idx` is provided, you must also provide "
                    "either `prev_neighbors` or `graph`."
                )
            else:
                prev_neighbors = _get_prev_neighbors(graph, reorder_idx)

    kwargs["reorder_idx"] = reorder_idx
    kwargs["inv_reorder_idx"] = inv_reorder_idx
    kwargs["prev_neighbors"] = prev_neighbors

    # Validity of the values will be checked by `_check_reorder_idx` in `RNNLayer`


def LSTMNet(*args, **kwargs):
    """Long short-term memory network."""

    _ensure_prev_neighbors(kwargs)
    return _LSTMNet(*args, **kwargs)


def GRUNet1D(*args, **kwargs):
    """Gated recurrent unit network. Only supports one previous neighbor at each site."""

    _ensure_prev_neighbors(kwargs)
    return _GRUNet1D(*args, **kwargs)
