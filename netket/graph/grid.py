# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

import itertools
from typing import List

from .graph import NetworkX

import numpy as _np
import networkx as _nx


class Grid(NetworkX):
    r"""A Grid lattice of d dimensions, and possibly different sizes of each dimension.
    Periodic boundary conditions can also be imposed"""

    def __init__(self, length: List, *, pbc: bool = True, color_edges: bool = False):
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
        if isinstance(pbc, list):
            self.pbc = pbc
        else:
            self.pbc = [pbc] * len(length)

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
        else:
            _nx.set_edge_attributes(graph, 0, name="color")

        newnames = {old: new for new, old in enumerate(graph.nodes)}
        graph = _nx.relabel_nodes(graph, newnames)

        super().__init__(graph)

    def __repr__(self):
        if all(self.pbc):
            pbc = True
        elif not any(self.pbc):
            pbc = False
        else:
            pbc = self.pbc
        return f"Grid(length={self.length}, pbc={pbc})"

    def periodic_translations(self) -> List[List[int]]:
        """
        Returns all permutations of lattice sites that correspond to translations
        along the grid directions with periodic boundary conditions.

        The periodic translations are a subset of the permutations returned by
        `self.automorphisms()`.
        """
        basis = [
            range(l) if is_per else range(1)
            for l, is_per in zip(self.length[::-1], self.pbc[::-1])
        ]

        translation_group = itertools.product(*basis)
        identity = _np.array(list(self.nodes())).reshape(*self.length[::-1])

        def translate(el, sites):
            for i, n in enumerate(el):
                sites = _np.roll(sites, shift=n, axis=i)
            return sites.ravel().tolist()

        return [translate(el, identity) for el in translation_group]

    def space_group(self, identity=None) -> List[List[int]]:
        """
        Returns all permutations of lattice sites that correspond to space group
        symmetry operations.

        The space group operations are a subset of the permutations returned by
        `self.automorphisms()`.
        """

        if not _np.any(identity):
            identity = _np.expand_dims(
                _np.array(list(self.nodes())).reshape(*self.length[::-1]), 0
            )
        ndim = len(self.length)

        dup_axes = []
        alr_dup = []

        for i, l in enumerate(self.length):
            if not (i in alr_dup):
                dups = []
                dups.append(ndim - i - 1)
                for j in range(i + 1, ndim):
                    if l == self.length[j]:
                        dups.append(ndim - j - 1)
                        alr_dup.append(j)
                dup_axes.append(dups)

        for i in range(ndim):
            if i == 0:
                perms = _np.concatenate((identity, _np.flip(identity, i + 1)), 0)
            else:
                perms = _np.concatenate((perms, _np.flip(perms, i + 1)), 0)

        for set_axes in dup_axes:

            set_axes = [p + 1 for p in set_axes]
            axis_perms = itertools.permutations(set_axes)

            iden = perms.copy()

            for i, axis_perm in enumerate(axis_perms):

                apply_perm = _np.arange(ndim + 1)
                apply_perm[set_axes] = axis_perm

                if i > 0:
                    perms = _np.concatenate((perms, iden.transpose(apply_perm)), 0)

        perms = _np.reshape(perms, [len(perms), -1])

        list_perms = []

        for perm in perms:
            list_perms.append(list(perm))

        return list_perms

    def lattice_group(self) -> List[List[int]]:
        """
        Returns all permutations of lattice sites that correspond to translation
        and space group symmetry operations. Translation is applied first, followed
        by reflection along lattice axes, followed by reflections along diagonals.

        The lattice permuations are a subset of the permutations returned by
        `self.automorphisms()`.
        """

        identity = _np.reshape(
            _np.array(self.periodic_translations()), [-1, *self.length[::-1]]
        )

        return self.space_group(identity)


def Hypercube(length: int, n_dim: int = 1, *, pbc: bool = True) -> Grid:
    r"""A hypercube lattice of side L in d dimensions.
    Periodic boundary conditions can also be imposed.

    Constructs a new ``Hypercube`` given its side length and dimension.

    Args:
        length: Side length of the hypercube; must always be >=1
        n_dim: Dimension of the hypercube; must be at least 1.
        pbc: If ``True`` then the constructed hypercube
            will have periodic boundary conditions, otherwise
            open boundary conditions are imposed.

    Examples:
         A 10x10x10 cubic lattice with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g = netket.graph.Hypercube(10, n_dim=3, pbc=True)
         >>> print(g.n_nodes)
         1000
    """
    length_vector = [length] * n_dim
    return Grid(length_vector, pbc=pbc)


def Square(length: int, *, pbc: bool = True) -> Grid:
    r"""A square lattice of side L.
    Periodic boundary conditions can also be imposed

    Constructs a new ``Square`` given its side length.

    Args:
        length: Side length of the square; must always be >=1
        pbc: If ``True`` then the constructed hypercube
            will have periodic boundary conditions, otherwise
            open boundary conditions are imposed.

    Examples:
        A 10x10 square lattice with periodic boundary conditions can be
        constructed as follows:

        >>> import netket
        >>> g=netket.graph.Square(10, pbc=True)
        >>> print(g.n_nodes)
        100
    """
    return Hypercube(length, n_dim=2, pbc=pbc)


def Chain(length: int, *, pbc: bool = True) -> Grid:
    r"""A chain of L sites.
    Periodic boundary conditions can also be imposed

    Constructs a new ``Chain`` given its length.

    Args:
        length: Length of the chain. It must always be >=1
        pbc: If ``True`` then the constructed chain
            will have periodic boundary conditions, otherwise
            open boundary conditions are imposed.

    Examples:
         A 10 site chain with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g = netket.graph.Chain(10, pbc=True)
         >>> print(g.n_nodes)
         10
    """
    return Hypercube(length, n_dim=1, pbc=pbc)
