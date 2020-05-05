from graph import Graph

import numpy as _np
import networkx as _nx


class Hypergrid(Graph):
    r"""A hypergrid lattice of d dimensions, and possibly different sizes of each dimension. 
	Periodic boundary conditions can also be imposed"""

    def __init__(self, length, pbc=True):
        """
		Constructs a new ``Grid`` given its length vector.

		Args:
			length: Side length of the hypergrid. It must be a list with integer components >= 1.
			pbc: If ``True```then the constructed hypergrid will have
				periodic boundary conditions, otherwise open boundary conditions
				are imposed.
		Examples:
		    A 5x10 lattice with periodic boundary conditions can be
		    constructed as follows:

		    >>> import netket
		    >>> g=netket.graph.Hypergrid(length=[5, 10], pbc=True)
		    >>> print(g.size)
		    50

		    Also, a 2x2x3 lattice with open boundary conditions can be constructed as follows:
		    >>> g=netket.graph.Hypergrid(length=[2,2,3], pbc=False)
		    >>> print(g.size)
		    12
		"""

        try:
            assert isinstance(length, list)
        except AssertionError:
            raise TypeError("length must be a list of integers >= 1")

        try:
            condition = [isinstance(x, int) and x >= 1 for x in length]
            if sum(condition) != len(length):
                raise TypeError
        except TypeError:
            raise TypeError("Components of length must be integers greater than 1")

        try:
            assert isinstance(pbc, bool)
        except AssertionError:
            raise TypeError("pbc must be a boolean")

        graph = _nx.generators.lattice.grid_graph(length, periodic=pbc)
        newnames = {old: new for new, old in enumerate(graph.nodes)}
        graph = _nx.relabel_nodes(graph, newnames)
        super().__init__(list(graph.edges))


class Hypercube(Hypergrid):
    r"""A hypercube lattice of side L in d dimensions. 
	Periodic boundary conditions can also be imposed"""

    def __init__(self, length, n_dim=1, pbc=True):
        """
		Constructs a new ``Hypercube`` given its side length and dimension.

		Args:
			length: Side length of the hypercube.
                 It must always be >=1,
                 but if ``pbc==True`` then the minimal
                 valid length is 3.
             n_dim: Dimension of the hypercube. It must be at least 1.
             pbc: If ``True`` then the constructed hypercube
                 will have periodic boundary conditions, otherwise
                 open boundary conditions are imposed.

		Examples:
             A 10x10x10 hypercubic lattice with periodic boundary conditions can be
             constructed as follows:

             >>> import netket
             >>> g=netket.graph.Hypercube(length=10,n_dim=3,pbc=True)
             >>> print(g.size)
             1000
		"""

        length_vector = [length] * n_dim

        super().__init__(length_vector, pbc)


class Square(Hypercube):
    r"""A square lattice of side L. 
	Periodic boundary conditions can also be imposed"""

    def __init__(self, length, pbc=True):
        """
		Constructs a new ``Square`` given its side length.

		Args:
			length: Side length of the square.
                 It must always be >=1,
                 but if ``pbc==True`` then the minimal
                 valid length is 3.
             pbc: If ``True`` then the constructed hypercube
                 will have periodic boundary conditions, otherwise
                 open boundary conditions are imposed.

		Examples:
             A 10x10 square lattice with periodic boundary conditions can be
             constructed as follows:

             >>> import netket
             >>> g=netket.graph.Square(length=10,pbc=True)
             >>> print(g.size)
             100
		"""
        super().__init__(length, n_dim=2, pbc=pbc)


class Chain(Hypercube):
    r"""A chain of L sites. 
	Periodic boundary conditions can also be imposed"""

    def __init__(self, length, pbc=True):
        """
		Constructs a new ``Chain`` given its length.

		Args:
			length: Length of the chain.
                 It must always be >=1,
                 but if ``pbc==True`` then the minimal
                 valid length is 3.
             pbc: If ``True`` then the constructed hypercube
                 will have periodic boundary conditions, otherwise
                 open boundary conditions are imposed.

		Examples:
             A 10 site lattice with periodic boundary conditions can be
             constructed as follows:

             >>> import netket
             >>> g=netket.graph.Square(length=10,pbc=True)
             >>> print(g.size)
             10
		"""
        super().__init__(length, n_dim=1, pbc=pbc)
