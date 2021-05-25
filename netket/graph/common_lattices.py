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

from functools import partial
from itertools import permutations
from typing import Sequence, Union, Optional, Tuple
import numpy as np

from .lattice import Lattice

from netket.utils.group import PointGroup, PGSymmetry, planar, Identity


def _perm_symm(perm: Tuple) -> PGSymmetry:
    n = len(perm)
    M = np.zeros((n, n))
    M[range(n), perm] = 1
    return PGSymmetry(M)


def _axis_reflection(axis: int, ndim: int) -> PGSymmetry:
    M = np.eye(ndim)
    M[axis, axis] = -1
    return PGSymmetry(M)


def _grid_point_group(extent: Sequence[int], pbc: Sequence[bool]) -> PointGroup:
    # axis permutations
    # * cannot exchange axes with open BC
    # * can exchange two PBC axes iff their lengths are the same
    axis_perm = []
    axes = np.arange(len(extent), dtype=int)
    obc = np.logical_not(pbc)
    ndim = len(extent)
    for perm in permutations(axes):
        if np.all(extent == extent[list(perm)]) and np.all(
            axes[obc] == axes[list(perm)][obc]
        ):
            axis_perm.append(_perm_symm(perm))
    result = PointGroup(axis_perm, ndim=ndim)
    # reflections across axes
    # can only do it across periodic axes
    for i in axes[pbc]:
        result = result @ PointGroup([Identity(), _axis_reflection(i, ndim)], ndim=ndim)
    result = result.elems
    result[0] = Identity()  # it would otherwise be an equivalent PGSymmetry
    return PointGroup(result, ndim=ndim)
    return result


def Grid(length: Sequence[int], *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    """
    Constructs a hypercubic lattice given its extent in all dimensions.

    Args:
        length: Side length of the lattice. It must be a list with integer components >= 1.
        pbc: If `True`, the grid will have periodic boundary conditions (PBC);
             if `False`, the grid will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

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
    length = np.asarray(length, dtype=int)
    ndim = len(length)
    if isinstance(pbc, bool):
        pbc = [pbc] * ndim
    return Lattice(
        basis_vectors=np.eye(ndim),
        extent=length,
        pbc=pbc,
        point_group=_grid_point_group(length, pbc),
    )


def Hypercube(length: int, n_dim: int = 1, *, pbc: bool = True) -> Lattice:
    r"""Constructs a hypercubic lattice with equal side length in all dimensions.
    Periodic boundary conditions can also be imposed.

    Args:
        length: Side length of the hypercube; must always be >=1
        n_dim: Dimension of the hypercube; must be at least 1.
        pbc: Whether the hypercube should have periodic boundary conditions (in all directions)

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


Square = partial(Hypercube, n_dim=2)
r"""Constructs a square lattice of side `length`
Periodic boundary conditions can also be imposed

Args:
    length: Side length of the square; must always be >=1
    pbc: Whether the square should have periodic boundary conditions (in both directions)

Examples:
    A 10x10 square lattice with periodic boundary conditions can be
    constructed as follows:

    >>> import netket
    >>> g=netket.graph.Square(10, pbc=True)
    >>> print(g.n_nodes)
    100
"""

Chain = partial(Hypercube, n_dim=1)
r"""Constructs a chain of `length` sites.
Periodic boundary conditions can also be imposed

Args:
    length: Length of the chain. It must always be >=1
    pbc: Whether the chain should have periodic boundary conditions

Examples:
    A 10 site chain with periodic boundary conditions can be
    constructed as follows:

    >>> import netket
    >>> g = netket.graph.Chain(10, pbc=True)
    >>> print(g.n_nodes)
    10
"""


def _hexagonal_general(
    extent, *, site_offsets=None, pbc: Union[bool, Sequence[bool]] = True
) -> Lattice:
    basis = [[1, 0], [0.5, 0.75 ** 0.5]]
    # determine if full point group is realised by the simulation box
    if isinstance(pbc, Sequence):
        all_pbc = pbc[0] and pbc[1]
    else:
        all_pbc = pbc
    point_group = planar.D(6) if all_pbc and extent[0] == extent[1] else None
    return Lattice(
        basis_vectors=basis,
        extent=extent,
        site_offsets=site_offsets,
        pbc=pbc,
        point_group=point_group,
    )


TriangularLattice = partial(_hexagonal_general, site_offsets=None)
r"""Constructs a triangular lattice of a given spatial extent.
Periodic boundary conditions can also be imposed
Sites are returned at the Bravais lattice points.

Args:
    extent: Number of unit cells along each direction, needs to be an array of length 2
    pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
         if `False`, the lattice will have open boundary conditions (OBC).
         This parameter can also be a list of booleans with same length as
         the parameter `length`, in which case each dimension will have
         PBC/OBC depending on the corresponding entry of `pbc`.

Example:
    Construct a triangular lattice with 3 × 3 unit cells:

    >>> from netket.graph import TriangularLattice
    >>> g = TriangularLattice(extent=[3, 3])
    >>> print(g.n_nodes)
    9
"""

HoneycombLattice = partial(
    _hexagonal_general, site_offsets=[[0.5, 0.5 / 3 ** 0.5], [1, 1 / 3 ** 0.5]]
)
r"""Constructs a honeycomb lattice of a given spatial extent.
Periodic boundary conditions can also be imposed.
Sites are returned at the 2b Wyckoff positions.

Args:
    extent: Number of unit cells along each direction, needs to be an array of length 2
    pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
         if `False`, the lattice will have open boundary conditions (OBC).
         This parameter can also be a list of booleans with same length as
         the parameter `length`, in which case each dimension will have
         PBC/OBC depending on the corresponding entry of `pbc`.

Example:
    Construct a honeycomb lattice with 3 × 3 unit cells:

    >>> from netket.graph import HoneycombLattice
    >>> g = HoneycombLattice(extent=[3, 3])
    >>> print(g.n_nodes)
    18
"""

KagomeLattice = partial(
    _hexagonal_general,
    site_offsets=[[0.5, 0], [0.25, 0.75 ** 0.5 / 2], [0.75, 0.75 ** 0.5 / 2]],
)
r"""Constructs a kagome lattice of a given spatial extent.
Periodic boundary conditions can also be imposed.
Sites are returned at the 3c Wyckoff positions.

Args:
    extent: Number of unit cells along each direction, needs to be an array of length 2
    pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
         if `False`, the lattice will have open boundary conditions (OBC).
         This parameter can also be a list of booleans with same length as
         the parameter `length`, in which case each dimension will have
         PBC/OBC depending on the corresponding entry of `pbc`.

Example:
    Construct a kagome lattice with 3 × 3 unit cells:

    >>> from netket.graph import KagomeLattice
    >>> g = KagomeLattice(extent=[3, 3])
    >>> print(g.n_nodes)
    27
"""
