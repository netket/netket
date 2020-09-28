from .graph import NetworkX

import numpy as _np
import networkx as _nx


class Grid(NetworkX):
    r"""A Grid lattice of d dimensions, and possibly different sizes of each dimension.
    Periodic boundary conditions can also be imposed"""

    def __init__(self, length, pbc=True, color_edges=False):
        """
        Constructs a new `Grid` given its length vector.

        Args:
            length: Side length of the Grid. It must be a list with integer components >= 1.
            pbc: If `True`, the grid will have periodic boundary conditions (PBC);
                if `False`, the grid will have open boundary conditions (OBC).
                This parameter can also be a list of booleans with same length as
                the parameter `length`, in which case each dimension will have
                PBC/OBC depending on the corresponding entry of `pbc`.
            color_edges: If `True`, the edges will be colored by their grid direction.
        Examples:
            A 5x10 lattice with periodic boundary conditions can be
            constructed as follows:

            >>> import netket
            >>> g=netket.graph.Grid(length=[5, 10], pbc=True)
            >>> print(g.n_nodes)
            50

            Also, a 2x2x3 lattice with open boundary conditions can be constructed as follows:
            >>> g=netket.graph.Grid(length=[2,2,3], pbc=False)
            >>> print(g.n_nodes)
            12
        """

        if not isinstance(length, list):
            raise TypeError("length must be a list of integers")

        try:
            condition = [isinstance(x, int) and x >= 1 for x in length]
            if sum(condition) != len(length):
                raise ValueError("Components of length must be integers greater than 1")
        except TypeError:
            raise ValueError("Components of length must be integers greater than 1")

        if not (isinstance(pbc, bool) or isinstance(pbc, list)):
            raise TypeError("pbc must be a boolean or list")
        if isinstance(pbc, list):
            if len(pbc) != len(length):
                raise ValueError("len(pbc) must be equal to len(length)")
            for l, p in zip(length, pbc):
                if l <= 2 and p:
                    raise ValueError("Directions with length <= 2 cannot have PBC")
            periodic = any(pbc)
        else:
            periodic = pbc

        self.length = length
        self.pbc = pbc

        graph = _nx.generators.lattice.grid_graph(length, periodic=periodic)

        # Remove unwanted periodic edges:
        if isinstance(pbc, list) and periodic:
            for e in graph.edges:
                for i, (l, is_per) in enumerate(zip(length[::-1], pbc[::-1])):
                    if l <= 2:
                        # Do not remove for short directions, because there is
                        # only one edge in that case.
                        continue
                    v1, v2 = sorted([e[0][i], e[1][i]])
                    if v1 == 0 and v2 == l - 1 and not is_per:
                        graph.remove_edge(*e)

        if color_edges:
            edges = {}
            for e in graph.edges:
                # color is the first (and only) dimension in which
                # the edge coordinates differ
                diff = _np.array(e[0]) - _np.array(e[1])
                color = int(_np.argwhere(diff[::-1] != 0))
                edges[e] = color
            _nx.set_edge_attributes(graph, edges, name="color")

        newnames = {old: new for new, old in enumerate(graph.nodes)}
        graph = _nx.relabel_nodes(graph, newnames)

        super().__init__(graph)

    def __repr__(self):
        return "Grid(length={}, pbc={})".format(self.length, self.pbc)


def Hypercube(length, n_dim=1, pbc=True):
    r"""A hypercube lattice of side L in d dimensions.
    Periodic boundary conditions can also be imposed.

    Constructs a new ``Hypercube`` given its side length and dimension.

    Args:
        length: Side length of the hypercube.
             It must always be >=1
         n_dim: Dimension of the hypercube. It must be at least 1.
         pbc: If ``True`` then the constructed hypercube
             will have periodic boundary conditions, otherwise
             open boundary conditions are imposed.

    Examples:
         A 10x10x10 hypercubic lattice with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g=netket.graph.Hypercube(length=10,n_dim=3,pbc=True)
         >>> print(g.n_nodes)
         1000
    """
    length_vector = [length] * n_dim
    return Grid(length_vector, pbc)


def Square(length, pbc=True):
    r"""A square lattice of side L.
    Periodic boundary conditions can also be imposed

    Constructs a new ``Square`` given its side length.

    Args:
        length: Side length of the square.
            It must always be >=1
        pbc: If ``True`` then the constructed hypercube
            will have periodic boundary conditions, otherwise
            open boundary conditions are imposed.

    Examples:
        A 10x10 square lattice with periodic boundary conditions can be
        constructed as follows:

        >>> import netket
        >>> g=netket.graph.Square(length=10,pbc=True)
        >>> print(g.n_nodes)
        100
    """
    return Hypercube(length, n_dim=2, pbc=pbc)


def Chain(length, pbc=True):
    r"""A chain of L sites.
    Periodic boundary conditions can also be imposed

    Constructs a new ``Chain`` given its length.

    Args:
      length: Length of the chain.
             It must always be >=1
         pbc: If ``True`` then the constructed hypercube
             will have periodic boundary conditions, otherwise
             open boundary conditions are imposed.

    Examples:
         A 10 site lattice with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g=netket.graph.Square(length=10,pbc=True)
         >>> print(g.n_nodes)
         10
    """
    return Hypercube(length, n_dim=1, pbc=pbc)
