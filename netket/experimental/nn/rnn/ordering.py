# Copyright 2023 The NetKet Authors - All rights reserved.
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

import numpy as np

from netket.graph import AbstractGraph, Lattice
from netket.utils import HashableArray
from netket.utils.types import Array


def check_reorder_idx(
    reorder_idx: HashableArray | None,
    inv_reorder_idx: HashableArray | None,
    prev_neighbors: HashableArray | None,
):
    """
    Check that the reordering indices determining the autoregressive order of an
    RNN are correctly declared.
    See :class:`netket.experimental.models.RNN` for details about the reordering indices.
    """
    if reorder_idx is None and inv_reorder_idx is None and prev_neighbors is None:
        # There is a faster code path for 1D RNN
        return

    if reorder_idx is None or inv_reorder_idx is None or prev_neighbors is None:
        raise ValueError(
            "`reorder_idx`, `inv_reorder_idx`, and `prev_neighbors` must be "
            "provided at the same time."
        )

    reorder_idx = np.asarray(reorder_idx)
    inv_reorder_idx = np.asarray(inv_reorder_idx)
    prev_neighbors = np.asarray(prev_neighbors)

    if reorder_idx.ndim != 1:
        raise ValueError("`reorder_idx` must be 1D.")
    if inv_reorder_idx.ndim != 1:
        raise ValueError("`inv_reorder_idx` must be 1D.")
    if prev_neighbors.ndim != 2:
        raise ValueError("`prev_neighbors` must be 2D.")

    V = reorder_idx.size
    if inv_reorder_idx.size != V:
        raise ValueError(
            "`reorder_idx` and `inv_reorder_idx` must have the same length."
        )
    if prev_neighbors.shape[0] != V:
        raise ValueError(
            "`reorder_idx` and `prev_neighbors` must have the same length."
        )

    idx = reorder_idx[inv_reorder_idx]
    if not np.array_equal(idx, np.arange(V)):
        raise ValueError("`inv_reorder_idx` is not the inverse of `reorder_idx`.")

    for i in range(V):
        n = prev_neighbors[i]
        for j in n:
            if j < -1 or j >= V:
                raise ValueError(f"Invaild neighbor {j} of site {i}")

        n = [j for j in n if j >= 0]
        if len(set(n)) != len(n):
            raise ValueError(f"Duplicate neighbor {n} of site {i}")

        for j in n:
            if reorder_idx[j] >= reorder_idx[i]:
                raise ValueError(f"Site {j} is not a previous neighbor of site {i}")


def ensure_prev_neighbors(
    *,
    reorder_idx: HashableArray | None = None,
    inv_reorder_idx: HashableArray | None = None,
    prev_neighbors: HashableArray | None = None,
    graph: AbstractGraph = None,
    check: bool = False,
) -> tuple[HashableArray, HashableArray, HashableArray]:
    """
    Deduce the missing arguments between *reorder_idx*,
    *inv_reorder_idx*, and *inv_reorder_idx* from the specified arguments.

    See :class:`netket.experimental.models.RNN` for details about the reordering indices.

    If no information on neighbors or graph is provided, assumes a 1D ordering.

    Args:
        reorder_idx: indices to transform the inputs from unordered to ordered.
            See :meth:`netket.models.AbstractARNN.reorder` for details
        inv_reorder_idx: indices to transform the inputs from ordered to unordered.
            See :meth:`netket.models.AbstractARNN.reorder` for details.
        prev_neighbors: previous neighbors of each site.
        graph: graph of the physical system, to deduce neighbors.
        check: check the validity of the provided values.
    """
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

    # Validity of the values will be checked by `_check_reorder_idx` in `RNNLayer`
    if check:
        check_reorder_idx(reorder_idx, inv_reorder_idx, prev_neighbors)

    return reorder_idx, inv_reorder_idx, prev_neighbors


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


def get_snake_inv_reorder_idx(graph: AbstractGraph) -> HashableArray:
    """
    A helper function to generate the inverse reorder indices in the snake order
    for a 2D graph.

    See :class:`netket.experimental.models.RNN` for details about the reordering indices.
    """
    V, L, M = _get_extent(graph)
    idx = np.arange(V, dtype=np.intp).reshape((L, M))
    idx[1::2, :] = idx[1::2, ::-1]
    idx = idx.flatten()
    idx = HashableArray(idx)
    return idx


def _get_inv_reorder_idx(graph: AbstractGraph) -> HashableArray:
    """
    A greedy algorithm to determine an autoregressive order with good locality.
    For any rectangular graph with OBC, it is the same as the snake order.
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
