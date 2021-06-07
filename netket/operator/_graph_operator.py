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

from textwrap import dedent

import numpy as np

from netket.utils.types import DType

from netket.graph import AbstractGraph
from netket.hilbert import AbstractHilbert

from ._local_operator import LocalOperator


def is_hermitian(a, rtol=1e-05, atol=1e-08):
    a = np.asarray(a)
    return np.allclose(a, a.T.conj(), rtol=rtol, atol=atol)


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
        dtype: DType = None,
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

        Examples:
         Constructs a ``GraphOperator`` operator for a 2D system.

         >>> import netket as nk
         >>> sigmax = [[0, 1], [1, 0]]
         >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
         >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
         >>> g = nk.graph.Graph(edges=edges)
         >>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)
         >>> op = nk.operator.GraphOperator(
         ... hi, site_ops=[sigmax], bond_ops=[mszsz], graph=g)
         >>> print(op)
         GraphOperator(dim=20, #acting_on=40 locations, constant=0, dtype=float64, graph=NetworkX(n_nodes=20))
        """

        if graph.n_nodes != hilbert.size:
            raise ValueError(
                dedent(
                    f"""The number of vertices in the graph ({graph.n_nodes})
                    must match the hilbert space size ({hilbert.size})"""
                )
            )

        # Ensure that at least one of SiteOps and BondOps was initialized
        if len(bond_ops) == 0 and len(site_ops) == 0:
            raise ValueError("Must input at least site_ops or bond_ops.")

        # Ensure that the bond operators are hermitian
        _non_hermitian = [i for (i, op) in enumerate(bond_ops) if not is_hermitian(op)]
        if len(_non_hermitian) > 0:
            raise ValueError(
                dedent(
                    f"""The bond operators at indices {_non_hermitian} are not
                            hermitian. `GraphOperator` only works with hermitian bond
                            operators because the ordering of edges in a graph is
                            arbitrary.
                            To build an hamiltonian with non-hermitian bonds please
                            sum `LocalOperator`s explicitly.

                            If you are using one of NetKet's built-in graphs, be care-
                            full not to rely on the ordering of the edges returned by
                            by `graph.edges()`, which is an undocumented, arbitrary
                            implementation detail and might change in future releases
                            of NetKet.
                            If you defined the list of vertices and edges yourself,
                            ignore this comment.

                            For Lattice-based graphs, it might be a good idea to iterate
                            lattice coordinates and use the relevant methods to convert
                            those to graph indices.
                            """
                )
            )

        # Create the local operator as the sum of all site and bond operators
        operators = []
        acting_on = []

        # Site operators
        if len(site_ops) > 0:
            for i in range(graph.n_nodes):
                for j, site_op in enumerate(site_ops):
                    operators.append(site_op)
                    acting_on.append([i])

        # Bond operators
        if len(bond_ops_colors) > 0:
            if len(bond_ops) != len(bond_ops_colors):
                raise ValueError(
                    dedent(
                        f"""The GraphOperator definition is inconsistent.
                    The sizes of bond_ops and bond_ops_color do not match:
                    len(bond_ops) != len(bond_ops_color)
                    ({len(bond_ops)} != {len(bond_ops_colors)}).
                    """
                    )
                )

            if len(bond_ops) > 0:
                #  Use edge_colors to populate operators
                for (u, v, color) in graph.edges(color=True):
                    edge = u, v
                    for c, bond_color in enumerate(bond_ops_colors):
                        if bond_color == color:
                            operators.append(bond_ops[c])
                            acting_on.append(edge)
        else:
            assert len(bond_ops) == 1

            for edge in graph.edges():
                operators.append(bond_ops[0])
                acting_on.append(edge)

        super().__init__(hilbert, operators, acting_on, dtype=dtype)
        self._graph = graph

    @property
    def graph(self) -> AbstractGraph:
        """The graph on which this Operator is defined"""
        return self._graph

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        _str = (
            f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, "
            f"constant={self.constant}, dtype={self.dtype}, graph={self.graph})"
        )
        return _str
