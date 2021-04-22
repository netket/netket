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
from dataclasses import dataclass
from functools import partial
from typing import Callable, List, Sequence, Tuple, Union

from plum import dispatch

import numpy as _np
import networkx as _nx

from netket.utils.semigroup import Element, Identity

from .symmetry import SymmGroup
from .graph import NetworkX


@dataclass(frozen=True)
class Translation(Element):
    shifts: Tuple[int]
    dims: Tuple[int]

    def __call__(self, sites):
        sites = sites.reshape(self.dims)
        for i, n in enumerate(self.shifts):
            sites = _np.roll(sites, shift=n, axis=i)
        return sites.ravel()

    def __repr__(self):
        return f"T{self.shifts}"


@dataclass(frozen=True)
class PlanarRotation(Element):

    num_quarter_rots: int
    axes: Tuple[int]
    dims: Tuple[int]

    def __call__(self, sites):
        sites = sites.reshape(self.dims)
        apply_perm = _np.arange(len(self.dims))
        apply_perm[list(self.axes)] = self.axes[::-1]
        for i in range(self.num_quarter_rots):
            sites = sites.transpose(apply_perm)
            sites = _np.flip(sites, self.axes[0])

        return sites.ravel()

    def __repr__(self):
        return f"Rot({self.num_quarter_rots / 2:.1f}π, axes={self.axes})"


@dataclass(frozen=True)
class Reflection(Element):
    axis: int
    dims: Tuple[int]

    def __call__(self, sites):
        sites = sites.reshape(self.dims)
        sites = _np.flip(sites, self.axis)
        return sites.ravel()

    def __repr__(self):
        return f"Ref(axis={self.axis})"


@dispatch
def product(a: Translation, b: Translation):
    if not a.dims == b.dims:
        raise ValueError("Incompatible translations")
    shifts = tuple(s1 + s2 for s1, s2 in zip(a.shifts, b.shifts))
    return Translation(shifts=shifts, dims=a.dims)


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

        graph = _nx.generators.lattice.grid_graph(length[::-1], periodic=periodic)

        # Remove unwanted periodic edges:
        if isinstance(pbc, list) and periodic:
            for e in graph.edges:
                for i, (l, is_per) in enumerate(zip(length, pbc)):
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
                color = int(_np.argwhere(diff != 0))
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

    def translations(
        self, dim: Union[int, Sequence[int]] = None, step: int = 1
    ) -> SymmGroup:
        """
        Returns all permutations of lattice sites that correspond to translations
        along the grid directions with periodic boundary conditions.

        The periodic translations are a subset of the permutations returned by
        `self.automorphisms()`.

        Arguments:
            dim: If set, only translations along `dim` will be returned. Can be a either
                a single dimension or a sequence of dimensions.
            step: Return translations by multiples of `step` sites (default: 1); should
                be a divisor of the length in the corresponding lattice dimension.
        """
        dims = tuple(self.length)
        if dim is None:
            basis = [
                range(0, l, step) if is_per else range(1)
                for l, is_per in zip(dims, self.pbc)
            ]
        else:
            if not isinstance(dim, Sequence):
                dim = (dim,)
            if not all(self.pbc[d] for d in dim):
                raise ValueError(
                    f"No translation symmetries in non-periodic dimensions"
                )
            basis = [
                range(0, l, step) if i in dim else range(1) for i, l in enumerate(dims)
            ]

        translations = itertools.product(*basis)
        next(translations)  # skip identity element here
        translations = [Translation(el, dims) for el in translations]

        return SymmGroup([Identity()] + translations, graph=self)

    def planar_rotation(self, axes: tuple = (0, 1)) -> SymmGroup:
        """
        Returns SymmGroup consisting of rotations about the origin in the plane defined by axes

        Arguments:
            axes: Axes that define the plane of rotation specified by dims.
        """

        dims = tuple(self.length)

        if not len(axes) == 2:
            raise ValueError(f"Plane is specified by two axes")
        if len(dims) < 2:
            raise ValueError(f"Rotations not defined for 1d systems")
        if _np.any(axes) > len(dims) - 1:
            raise ValueError(f"Axis specified not in dims")

        if self.length[axes[0]] == self.length[axes[1]]:
            basis = (range(0, 4), [axes])
        else:
            basis = (range(0, 4, 2), [axes])

        rotations = itertools.product(*basis)
        next(rotations)

        rotations = [PlanarRotation(num, ax, dims) for (num, ax) in rotations]

        return SymmGroup([Identity()] + rotations, graph=self)

    def axis_reflection(self, axis: int = 0) -> SymmGroup:
        """
        Returns SymmGroup consisting of identity and the lattice
        reflected about the hyperplane axis = 0

        Arguments:
            axis: Axis to be reflected about
        """
        if abs(axis) > len(self.length) - 1:
            raise ValueError(f"Axis specified not in dims")

        dims = tuple(self.length)
        return SymmGroup([Identity(), Reflection(axis, dims)], graph=self)

    def rotations(self, *, remove_duplicates: bool = True) -> SymmGroup:
        """
        Returns all possible rotation symmetries of the lattice.

        The rotations are a subset of the permutations returned by
        `self.automorphisms()`.

        Arguments:
            period: Period of the rotations; should be a divisor of 4.
            remove_duplicates: Only include unique rotations.
        """
        axes = itertools.combinations(range(len(self.length)), 2)
        group = SymmGroup([Identity()], graph=self)

        for axs in axes:
            group = group @ self.planar_rotation(axs)

        if remove_duplicates:
            return group.remove_duplicates()
        else:
            return group

    def space_group(self) -> SymmGroup:
        """
        Returns the full space group of the lattice.

        The space group is a subset of the permutations returned by
        `self.automorphisms()`.

        Arguments:
            remove_duplicates: Only include unique space group elements.
        """
        return self.rotations() @ self.axis_reflection()

    def lattice_group(self) -> SymmGroup:
        """
        Returns the full lattice symmetry group consisting of rotations, reflections, and periodic translation.

        The lattice group is a subset of the permutations returned by
        `self.automorphisms()`.

        """
        return self.translations() @ self.space_group()


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
