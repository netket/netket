# Copyright 2021 The NetKet Authors - All rights reserved.
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

from collections.abc import Sequence
from typing import TypeVar as _TypeVar

import numpy as np

from netket.graph import AbstractGraph
from netket.hilbert import AbstractHilbert
from netket.utils.types import DType

from ._graph_operator import GraphOperator
from ._local_operator import (
    LocalOperator as _LocalOperator,
    LocalOperatorBase as _LocalOperatorBase,
)

_LocalOperatorT = _TypeVar("_LocalOperatorT", bound=_LocalOperatorBase)


def _is_sublattice_rotation(n_nodes: int, edges, flipped) -> bool:
    """
    Checks whether flipping the sign of the exchange term on the edges marked
    by ``flipped`` is a sublattice rotation, i.e. whether the sites can be split
    into two sublattices A and B such that the flipped edges are exactly the ones
    connecting A to B. This holds if and only if every cycle of the graph
    contains an even number of flipped edges.
    """
    adjacency = [[] for _ in range(n_nodes)]
    for (i, j), f in zip(edges, flipped):
        adjacency[i].append((j, f))
        adjacency[j].append((i, f))

    sublattice = [None] * n_nodes
    for root in range(n_nodes):
        if sublattice[root] is not None:
            continue
        sublattice[root] = False
        stack = [root]
        while stack:
            i = stack.pop()
            for j, f in adjacency[i]:
                expected = sublattice[i] ^ f
                if sublattice[j] is None:
                    sublattice[j] = expected
                    stack.append(j)
                elif sublattice[j] != expected:
                    return False
    return True


def Heisenberg(
    hilbert: AbstractHilbert,
    graph: AbstractGraph,
    J: float | Sequence[float] = 1.0,
    sign_rule: None | bool | Sequence[bool] = None,
    dtype: DType | None = None,
    *,
    acting_on_subspace: None | list[int] | int = None,
    cls: type[_LocalOperatorT] = _LocalOperator,
) -> _LocalOperatorT:
    """
    Constructs an Heisenberg operator given a hilbert space and a graph providing the
    connectivity of the lattice.

    Args:
        hilbert: Hilbert space the operator acts on.
        graph: The graph upon which this hamiltonian is defined.
        J: The strength of the coupling. Default is 1.
           Can pass a sequence of coupling strengths with coloured graphs:
           edges of colour n will have coupling strength J[n]
        sign_rule: If True, Marshal's sign rule will be used. On a bipartite
            lattice, this corresponds to a basis change flipping the Sz direction
            at every odd site of the lattice. For non-bipartite lattices, the
            sign rule cannot be applied. Defaults to True if the lattice is
            bipartite, False otherwise.
            If a sequence of coupling strengths is passed, defaults to False
            and a matching sequence of sign_rule must be specified to override it.
            In that case, the sign is flipped on all edges of the colors with
            sign_rule=True, and this is only allowed if it is a basis change,
            that is if the sites can be split into two sublattices such that
            those edges connect different sublattices, and the edges of the
            other colors connect sites of the same sublattice. For example,
            on a J1-J2 lattice ``sign_rule=[True, False]`` is allowed, while
            ``sign_rule=[False, True]`` is not.
        dtype: Data type of the matrix elements.
        acting_on_subspace: Specifies the mapping between nodes of the graph and
            Hilbert space sites, so that graph node :code:`i ∈ [0, ..., graph.n_nodes - 1]`,
            corresponds to :code:`acting_on_subspace[i] ∈ [0, ..., hilbert.n_sites]`.
            Must be a list of length `graph.n_nodes`. Passing a single integer :code:`start`
            is equivalent to :code:`[start, ..., start + graph.n_nodes - 1]`.

    Examples:
     Constructs a ``Heisenberg`` operator for a 1D system.

        >>> import netket as nk
        >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
        >>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
        >>> op = nk.operator.Heisenberg(hilbert=hi, graph=g)
        >>> print(op)
        LocalOperatorJax(dim=20, #acting_on=20 locations, constant=0.0, dtype=float64)
    """
    if isinstance(J, Sequence):
        # check that the number of Js matches the number of colours
        assert len(J) == max(graph.edge_colors) + 1

        if sign_rule is None:
            sign_rule = [False] * len(J)
        else:
            assert len(sign_rule) == len(J)
            # The sign rule must be the same sublattice rotation for all colors
            # at once: checking that every color is bipartite on its own is not
            # enough (e.g. sign_rule=[False, True] on a J1-J2 lattice).
            # Colors with J=0 do not contribute to the operator and are ignored.
            edges = [
                (i, j, c) for i, j, c in graph.edges(return_color=True) if J[c] != 0
            ]
            if not _is_sublattice_rotation(
                graph.n_nodes,
                [(i, j) for i, j, _ in edges],
                [bool(sign_rule[c]) for _, _, c in edges],
            ):
                raise ValueError(
                    f"sign_rule={list(sign_rule)} does not correspond to a "
                    "sublattice rotation of this graph, so it would change the "
                    "spectrum of the Hamiltonian. The sign rule can only be "
                    "applied if the sites can be split into two sublattices "
                    "such that the edges of the colors with sign_rule=True "
                    "connect different sublattices, and those of the colors "
                    "with sign_rule=False connect sites of the same one."
                )
    else:
        if sign_rule is None:
            sign_rule = graph.is_bipartite()
        elif sign_rule and not graph.is_bipartite():
            raise ValueError("sign_rule=True specified for a non-bipartite lattice")

    sz_sz = np.array(
        [
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1],
        ]
    )
    exchange = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 2, 0],
            [0, 2, 0, 0],
            [0, 0, 0, 0],
        ]
    )

    if isinstance(J, Sequence):
        bond_ops = [
            J[i] * (sz_sz - exchange if sign_rule[i] else sz_sz + exchange)
            for i in range(len(J))
        ]
        bond_ops_colors = list(range(len(J)))
    else:
        bond_ops = [J * (sz_sz - exchange if sign_rule else sz_sz + exchange)]
        bond_ops_colors = []

    return GraphOperator(
        hilbert,
        graph,
        bond_ops=bond_ops,
        bond_ops_colors=bond_ops_colors,
        dtype=dtype,
        acting_on_subspace=acting_on_subspace,
        cls=cls,
    )
