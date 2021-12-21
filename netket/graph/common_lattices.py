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

from itertools import permutations
from typing import Sequence, Union, Tuple
import numpy as np
import warnings

from .lattice import Lattice

from netket.utils.group import PointGroup, PGSymmetry, planar, cubic, Identity


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
    # can exchange two axes iff they have the same kind of BC and length
    # represent open BC by setting kind[i] = -extent[i], so just have to
    # match these
    axis_perm = []
    axes = np.arange(len(extent), dtype=int)
    extent = np.asarray(extent, dtype=int)
    kind = np.where(pbc, extent, -extent)
    ndim = len(extent)
    for perm in permutations(axes):
        if np.all(kind == kind[list(perm)]):
            axis_perm.append(_perm_symm(perm))
    result = PointGroup(axis_perm, ndim=ndim)
    # reflections across axes and setting the origin
    # OBC axes are only symmetric w.r.t. their midpoint, (extent[i]-1)/2
    origin = []
    for i in axes:
        result = result @ PointGroup([Identity(), _axis_reflection(i, ndim)], ndim=ndim)
        origin.append(0 if pbc[i] else (extent[i] - 1) / 2)
    result = result.elems
    result[0] = Identity()  # it would otherwise be an equivalent PGSymmetry
    return PointGroup(result, ndim=ndim).change_origin(origin)


def Grid(
    extent: Sequence[int] = None,
    *,
    length: Sequence[int] = None,
    pbc: Union[bool, Sequence[bool]] = True,
    **kwargs,
) -> Lattice:
    """
    Constructs a hypercubic lattice given its extent in all dimensions.

    Args:
        extent: Size of the lattice along each dimension. It must be a list with
                integer components >= 1.
        pbc: If `True`, the grid will have periodic boundary conditions (PBC);
             if `False`, the grid will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
        Construct a 5x10 square lattice with periodic boundary conditions:

        >>> import netket
        >>> g=netket.graph.Grid(extent=[5, 10], pbc=True)
        >>> print(g.n_nodes)
        50

        Construct a 2x2x3 cubic lattice with open boundary conditions:

        >>> g=netket.graph.Grid(extent=[2,2,3], pbc=False)
        >>> print(g.n_nodes)
        12
    """
    if extent is None:
        if length is None:
            raise TypeError("Required argument 'extent' missing")
        else:
            warnings.warn(
                "'length' is deprecated and may be removed in future versions, "
                "use 'extent' instead",
                FutureWarning,
            )
            extent = np.asarray(length, dtype=int)
    else:
        if length is not None:
            raise TypeError(
                "'length' is a deprecated alias of 'extent', do not supply both"
            )
        else:
            extent = np.asarray(extent, dtype=int)

    ndim = len(extent)
    if isinstance(pbc, bool):
        pbc = [pbc] * ndim
    return Lattice(
        basis_vectors=np.eye(ndim),
        extent=extent,
        pbc=pbc,
        point_group=lambda: _grid_point_group(extent, pbc),
        **kwargs,
    )


def Hypercube(length: int, n_dim: int = 1, *, pbc: bool = True, **kwargs) -> Lattice:
    r"""Constructs a hypercubic lattice with equal side length in all dimensions.
    Periodic boundary conditions can also be imposed.

    Args:
        length: Side length of the hypercube; must always be >=1
        n_dim: Dimension of the hypercube; must be at least 1.
        pbc: Whether the hypercube should have periodic boundary conditions
            (in all directions)
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
         A 10x10x10 cubic lattice with periodic boundary conditions can be
         constructed as follows:

         >>> import netket
         >>> g = netket.graph.Hypercube(10, n_dim=3, pbc=True)
         >>> print(g.n_nodes)
         1000
    """
    if not isinstance(length, int) or length <= 0:
        raise TypeError("Argument `length` must be a positive integer")
    length_vector = [length] * n_dim
    return Grid(length_vector, pbc=pbc, **kwargs)


def Cube(length: int, *, pbc: bool = True, **kwargs) -> Lattice:
    """Constructs a cubic lattice of side `length`
    Periodic boundary conditions can also be imposed

    Args:
        length: Side length of the cube; must always be >=1
        pbc: Whether the cube should have periodic boundary conditions
            (in all directions)
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
        A 10×10×10 cubic lattice with periodic boundary conditions can be
        constructed as follows:

        >>> import netket
        >>> g=netket.graph.Cube(10, pbc=True)
        >>> print(g.n_nodes)
        1000
    """
    return Hypercube(length, pbc=pbc, n_dim=3, **kwargs)


def Square(length: int, *, pbc: bool = True, **kwargs) -> Lattice:
    """Constructs a square lattice of side `length`
    Periodic boundary conditions can also be imposed

    Args:
        length: Side length of the square; must always be >=1
        pbc: Whether the square should have periodic boundary
            conditions (in both directions)
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
        A 10x10 square lattice with periodic boundary conditions can be
        constructed as follows:

        >>> import netket
        >>> g=netket.graph.Square(10, pbc=True)
        >>> print(g.n_nodes)
        100
    """
    return Hypercube(length, pbc=pbc, n_dim=2, **kwargs)


def Chain(length: int, *, pbc: bool = True, **kwargs) -> Lattice:
    r"""Constructs a chain of `length` sites.
    Periodic boundary conditions can also be imposed

    Args:
        length: Length of the chain. It must always be >=1
        pbc: Whether the chain should have periodic boundary conditions
        kwargs: Additional keyword arguments are passed on to the constructor of
            :ref:`netket.graph.Lattice`.

    Examples:
        A 10 site chain with periodic boundary conditions can be
        constructed as follows:

        >>> import netket
        >>> g = netket.graph.Chain(10, pbc=True)
        >>> print(g.n_nodes)
        10
    """
    return Hypercube(length, pbc=pbc, n_dim=1, **kwargs)


def BCC(extent: Sequence[int], *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    """Constructs a BCC lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed
    Sites are returned at the Bravais lattice points.

    Arguments:
        extent: Number of primitive unit cells along each direction, needs to be
            an array of length 3
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a BCC lattice with 3×3×3 primitive unit cells:

        >>> from netket.graph import BCC
        >>> g = BCC(extent=[3,3,3])
        >>> print(g.n_nodes)
        27
    """
    basis = [[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]
    # determine if full point group is realised by the simulation box
    point_group = cubic.Oh() if np.all(pbc) and len(set(extent)) == 1 else None
    return Lattice(basis_vectors=basis, extent=extent, pbc=pbc, point_group=point_group)


def FCC(extent: Sequence[int], *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    """Constructs an FCC lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed
    Sites are returned at the Bravais lattice points.

    Arguments:
        extent: Number of primitive unit cells along each direction, needs
            to be an array of length 3
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct an FCC lattice with 3×3×3 primitive unit cells:

        >>> from netket.graph import FCC
        >>> g = FCC(extent=[3,3,3])
        >>> print(g.n_nodes)
        27
    """
    basis = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    # determine if full point group is realised by the simulation box
    point_group = cubic.Oh() if np.all(pbc) and len(set(extent)) == 1 else None
    return Lattice(basis_vectors=basis, extent=extent, pbc=pbc, point_group=point_group)


def Diamond(
    extent: Sequence[int], *, pbc: Union[bool, Sequence[bool]] = True
) -> Lattice:
    """Constructs a diamond lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed.

    Sites are returned at the 8a Wyckoff positions of the FCC lattice
    ([000], [1/4,1/4,1/4], and translations thereof).

    Arguments:
        extent: Number of primitive unit cells along each direction, needs to
            be an array of length 3
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a diamond lattice with 3×3×3 primitive unit cells:

        >>> from netket.graph import Diamond
        >>> g = Diamond(extent=[3,3,3])
        >>> print(g.n_nodes)
        54
    """
    basis = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    sites = [[0, 0, 0], [0.25, 0.25, 0.25]]
    # determine if full point group is realised by the simulation box
    point_group = cubic.Fd3m() if np.all(pbc) and len(set(extent)) == 1 else None
    return Lattice(
        basis_vectors=basis,
        site_offsets=sites,
        extent=extent,
        pbc=pbc,
        point_group=point_group,
    )


def Pyrochlore(
    extent: Sequence[int], *, pbc: Union[bool, Sequence[bool]] = True
) -> Lattice:
    """Constructs a pyrochlore lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed.

    Sites are returned at the 16c Wyckoff positions of the FCC lattice
    ([111]/8, [1 -1 -1]/8, [-1 1 -1]/8, [-1 -1 1]/8, and translations thereof).

    Arguments:
        extent: Number of primitive unit cells along each direction, needs to be
            an array of length 3
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
            if `False`, the lattice will have open boundary conditions (OBC).
            This parameter can also be a list of booleans with same length as
            the parameter `length`, in which case each dimension will have
            PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a pyrochlore lattice with 3×3×3 primitive unit cells:

        >>> from netket.graph import Pyrochlore
        >>> g = Pyrochlore(extent=[3,3,3])
        >>> print(g.n_nodes)
        108
    """
    basis = [[0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    sites = np.array([[1, 1, 1], [1, 3, 3], [3, 1, 3], [3, 3, 1]]) / 8
    # determine if full point group is realised by the simulation box
    point_group = cubic.Fd3m() if np.all(pbc) and len(set(extent)) == 1 else None
    return Lattice(
        basis_vectors=basis,
        site_offsets=sites,
        extent=extent,
        pbc=pbc,
        point_group=point_group,
    )


def _hexagonal_general(
    extent, *, site_offsets=None, pbc: Union[bool, Sequence[bool]] = True
) -> Lattice:
    basis = [[1, 0], [0.5, 0.75 ** 0.5]]
    # determine if full point group is realised by the simulation box
    point_group = planar.D(6) if np.all(pbc) and extent[0] == extent[1] else None
    return Lattice(
        basis_vectors=basis,
        extent=extent,
        site_offsets=site_offsets,
        pbc=pbc,
        point_group=point_group,
    )


def Triangular(extent, *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    r"""Constructs a triangular lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed
    Sites are returned at the Bravais lattice points.

    Arguments:
        extent: Number of unit cells along each direction, needs to be an array
            of length 2
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a triangular lattice with 3 × 3 unit cells:

        >>> from netket.graph import Triangular
        >>> g = Triangular(extent=[3, 3])
        >>> print(g.n_nodes)
        9
    """
    return _hexagonal_general(extent, site_offsets=None, pbc=pbc)


def Honeycomb(extent, *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    r"""Constructs a honeycomb lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed.
    Sites are returned at the 2b Wyckoff positions.

    Arguments:
        extent: Number of unit cells along each direction, needs to be an array
            of length 2
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a honeycomb lattice with 3 × 3 unit cells:

        >>> from netket.graph import Honeycomb
        >>> g = Honeycomb(extent=[3, 3])
        >>> print(g.n_nodes)
        18
    """
    return _hexagonal_general(
        extent, site_offsets=[[0.5, 0.5 / 3 ** 0.5], [1, 1 / 3 ** 0.5]], pbc=pbc
    )


def Kagome(extent, *, pbc: Union[bool, Sequence[bool]] = True) -> Lattice:
    r"""Constructs a kagome lattice of a given spatial extent.
    Periodic boundary conditions can also be imposed.
    Sites are returned at the 3c Wyckoff positions.

    Arguments:
        extent: Number of unit cells along each direction, needs to be an array
            of length 2
        pbc: If `True`, the lattice will have periodic boundary conditions (PBC);
             if `False`, the lattice will have open boundary conditions (OBC).
             This parameter can also be a list of booleans with same length as
             the parameter `length`, in which case each dimension will have
             PBC/OBC depending on the corresponding entry of `pbc`.

    Example:
        Construct a kagome lattice with 3 × 3 unit cells:

        >>> from netket.graph import Kagome
        >>> g = Kagome(extent=[3, 3])
        >>> print(g.n_nodes)
        27
    """
    return _hexagonal_general(
        extent,
        site_offsets=[[0.5, 0], [0.25, 0.75 ** 0.5 / 2], [0.75, 0.75 ** 0.5 / 2]],
        pbc=pbc,
    )
