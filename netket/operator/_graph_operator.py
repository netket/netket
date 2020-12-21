from ._abstract_operator import AbstractOperator
from ._local_operator import LocalOperator

import numpy as _np
from numba import jit


class GraphOperator(LocalOperator):
    def __init__(self, hilbert, graph, site_ops=[], bond_ops=[], bond_ops_colors=[]):
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
             bond operator acts on. The defualt is an empty list.

        Examples:
         Constructs a ``GraphOperator`` operator for a 2D system.

         >>> import netket as nk
         >>> sigmax = [[0, 1], [1, 0]]
         >>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
         >>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
         ... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
         ... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
         >>> g = nk.graph.CustomGraph(edges=edges)
         >>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
         >>> op = nk.operator.GraphOperator(
         ... hi, site_ops=[sigmax], bond_ops=[mszsz])
         >>> print(op.hilbert.size)
         20
        """

        assert (
            graph.n_nodes == hilbert.size
        ), "The size of the graph must match the hilbert space"

        self.graph = graph
        size = graph.n_nodes

        # Create the local operator as the sum of all site and bond operators

        # Ensure that at least one of SiteOps and BondOps was initialized
        if len(bond_ops) == 0 and len(site_ops) == 0:
            raise InvalidInputError("Must input at least site_ops or bond_ops.")

        super().__init__(hilbert)

        # Site operators
        if len(site_ops) > 0:
            for i in range(size):
                for j, site_op in enumerate(site_ops):
                    self += LocalOperator(hilbert, site_op, [i])

        # Bond operators
        if len(bond_ops_colors) > 0:
            if len(bond_ops) != len(bond_ops_colors):
                raise InvalidInputError(
                    """The GraphHamiltonian definition is inconsistent.
                    The sizes of bond_ops and bond_ops_colors do not match."""
                )

            if len(bond_ops) > 0:
                #  Use edge_colors to populate operators
                for (u, v, color) in graph.edges(color=True):
                    edge = u, v
                    for c, bond_color in enumerate(bond_ops_colors):
                        if bond_color == color:
                            self += LocalOperator(hilbert, bond_ops[c], edge)
        else:
            assert len(bond_ops) == 1

            for edge in graph.edges():
                self += LocalOperator(hilbert, bond_ops[0], edge)
