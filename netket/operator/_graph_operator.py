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

from typing import Optional, Union

from netket.utils.types import DType

from netket.graph import AbstractGraph
from netket.hilbert import AbstractHilbert

from ._local_operator import LocalOperator


def check_acting_on_subspace(acting_on_subspace, hilbert, graph):
    """Check `acting_on_subspace` argument used by various operators."""
    if acting_on_subspace is None:
        acting_on_subspace = list(range(hilbert.size))
    elif isinstance(acting_on_subspace, int):
        start = acting_on_subspace
        acting_on_subspace = [start + i for i in range(graph.n_nodes)]
    elif isinstance(acting_on_subspace, list):
        if len(acting_on_subspace) != graph.n_nodes:
            raise ValueError(
                "acting_on_subspace must be a list of length graph.n_nodes"
            )
    else:
        raise TypeError("acting_on_subspace must be a list or single integer")

    return acting_on_subspace


class GraphOperator(LocalOperator):
    r"""
    A graph-based quantum operator. In its simplest terms, this is the sum of
    local operators living on the edge of an arbitrary graph.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        graph: AbstractGraph,
        site_ops=[],
        bond_ops=[],
        bond_ops_colors=[],
        dtype: Optional[DType] = None,
        *,
        acting_on_subspace: Union[None, list[int], int] = None,
    ):
        r"""
        A graph-based quantum operator. In its simplest terms, this is the sum of
        local operators living on the edge of an arbitrary graph.

        A ``GraphOperator`` is constructed giving a hilbert space and either a
        list of operators acting on sites or a list acting on the bonds.
        Users can specify the color of the bond that an operator acts on, if
        desired. If none are specified, the bond operators act on all edges.

        Args:
         hilbert: Hilbert space the operator acts on.
         graph: The graph whose vertices and edges are considered to construct the
                operator
         site_ops: A list of operators in matrix form that act
                on the nodes of the graph.
                The default is an empty list. Note that if no site_ops are
                specified, the user must give a list of bond operators.
         bond_ops: A list of operators that act on the edges of the graph.
             The default is None. Note that if no bond_ops are
             specified, the user must give a list of site operators.
         bond_ops_colors: A list of edge colors, specifying the color each
             bond operator acts on. The default is an empty list.
         dtype: Data type of the matrix elements.
         acting_on_subspace: Specifies the mapping between nodes of the graph and
            Hilbert space sites, so that graph node :code:`i ∈ [0, ..., graph.n_nodes]`,
            corresponds to :code:`acting_on_subspace[i] ∈ [0, ..., hilbert.n_sites]`.
            Must be a list of length `graph.n_nodes`. Passing a single integer :code:`start`
            is equivalent to :code:`[start, ..., start + graph.n_nodes - 1]`.

        Examples:
         Constructs a ``GraphOperator`` operator for a 2D system.

         >>> import netket as nk
         >>> sigmax = [[0, 1], [1, 0]]
         >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
         >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
         >>> g = nk.graph.Graph(edges=edges)
         >>> hi = nk.hilbert.Spin(0.5, N=g.n_nodes)
         >>> op = nk.operator.GraphOperator(
         ... hi, site_ops=[sigmax], bond_ops=[mszsz], graph=g)
         >>> print(op)
         GraphOperator(dim=20, #acting_on=40 locations, constant=0.0, dtype=float64, graph=Graph(n_nodes=20, n_edges=20))
        """
        acting_on_subspace = check_acting_on_subspace(
            acting_on_subspace, hilbert, graph
        )
        self._acting_on_subspace = acting_on_subspace

        # Ensure that at least one of SiteOps and BondOps was initialized
        if len(bond_ops) == 0 and len(site_ops) == 0:
            raise ValueError("Must input at least site_ops or bond_ops.")

        # Create the local operator as the sum of all site and bond operators
        operators = []
        acting_on = []

        # Site operators
        if len(site_ops) > 0:
            for i in range(graph.n_nodes):
                for site_op in site_ops:
                    i_prime = acting_on_subspace[i]
                    operators.append(site_op)
                    acting_on.append([i_prime])

        # Bond operators
        if len(bond_ops_colors) > 0:
            if len(bond_ops) != len(bond_ops_colors):
                raise ValueError(
                    """The GraphHamiltonian definition is inconsistent.
                    The sizes of bond_ops and bond_ops_colors do not match."""
                )

            if len(bond_ops) > 0:
                #  Use edge_colors to populate operators
                for u, v, color in graph.edges(return_color=True):
                    u, v = acting_on_subspace[u], acting_on_subspace[v]
                    for c, bond_color in enumerate(bond_ops_colors):
                        if bond_color == color:
                            operators.append(bond_ops[c])
                            acting_on.append([u, v])
        else:
            assert len(bond_ops) == 1

            for u, v in graph.edges():
                u, v = acting_on_subspace[u], acting_on_subspace[v]
                operators.append(bond_ops[0])
                acting_on.append([u, v])

        super().__init__(hilbert, operators, acting_on, dtype=dtype)
        self._graph = graph

    @property
    def graph(self) -> AbstractGraph:
        """The graph on which this Operator is defined"""
        return self._graph

    @property
    def acting_on_subspace(self):
        """
        Mapping between nodes of the graph and Hilbert space sites as given in
        the constructor.
        """
        return self._acting_on_subspace

    def copy(self, *, dtype: Optional[DType] = None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype
        """
        return super().copy(dtype=dtype, _cls=LocalOperator)

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, constant={self.constant}, dtype={self.dtype}, graph={self.graph})"
