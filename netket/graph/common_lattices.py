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

from typing import Sequence, Union, Optional
import numpy as np

from .lattice import Lattice

from netket.utils.group import PointGroup, cubic


def Grid(
    length: Sequence[int],
    *,
    pbc: Union[bool, Sequence[bool]] = True,
    point_group: Optional[PointGroup] = None,
) -> Lattice:
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
    return Lattice(
        basis_vectors=np.eye(ndim), extent=length, pbc=pbc, point_group=point_group
    )


def Hypercube(length: int, n_dim: int = 1, *, pbc: bool = True) -> Lattice:
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
    return Grid(length_vector, pbc=pbc, point_group=cubic.hypercubic(n_dim))


def Square(length: int, *, pbc: bool = True) -> Lattice:
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


def Chain(length: int, *, pbc: bool = True) -> Lattice:
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
