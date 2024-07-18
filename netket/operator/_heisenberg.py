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

import numpy as np

from netket.graph import AbstractGraph, Graph
from netket.hilbert import AbstractHilbert
from netket.utils.types import DType

from ._graph_operator import GraphOperator


class Heisenberg(GraphOperator):
    r"""
    The Heisenberg hamiltonian on a lattice.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        J: float | Sequence[float] = 1.0,
        sign_rule: None | bool | Sequence[bool] = None,
        dtype: DType | None = None,
        *,
        acting_on_subspace: None | list[int] | int = None,
    ):
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
                and a matching sequence of sign_rule must be specified to override it
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
            Heisenberg(J=1.0, sign_rule=True; dim=20)
        """
        if isinstance(J, Sequence):
            # check that the number of Js matches the number of colours
            assert len(J) == max(graph.edge_colors) + 1

            if sign_rule is None:
                sign_rule = [False] * len(J)
            else:
                assert len(sign_rule) == len(J)
                for i in range(len(J)):
                    subgraph = Graph(edges=graph.edges(filter_color=i))
                    if sign_rule[i] and not subgraph.is_bipartite():
                        raise ValueError(
                            "sign_rule=True specified for a non-bipartite lattice"
                        )
        else:
            if sign_rule is None:
                sign_rule = graph.is_bipartite()
            elif sign_rule and not graph.is_bipartite():
                raise ValueError("sign_rule=True specified for a non-bipartite lattice")

        self._J = J
        self._sign_rule = sign_rule

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

        super().__init__(
            hilbert,
            graph,
            bond_ops=bond_ops,
            bond_ops_colors=bond_ops_colors,
            dtype=dtype,
            acting_on_subspace=acting_on_subspace,
        )

    @property
    def J(self) -> float:
        """The coupling strength."""
        return self._J

    @property
    def uses_sign_rule(self):
        return self._sign_rule

    def __repr__(self):
        return f"Heisenberg(J={self._J}, sign_rule={self._sign_rule}; dim={self.hilbert.size})"
