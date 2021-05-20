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

from dataclasses import dataclass
import itertools
from itertools import product
from math import pi
from netket.utils.types import Array
from typing import Dict, Sequence, Tuple, Union, Optional
import warnings

import networkx as _nx
import numpy as _np
from scipy.spatial import cKDTree
from scipy.sparse import find, triu

from netket.utils.deprecation import deprecated as _deprecated, warn_deprecation
from netket.utils.semigroup import Identity, Element, PermutationGroup
from netket.utils import HashableArray

from .graph import NetworkX

tol_digits = 5
cutoff_tol = _np.power(10.0, -tol_digits)
"""Tolerance for the maximum distance cutoff when computing the sparse distance matrix.
This is necessary because of floating-point errors when computing the distance in non-trivial
lattices.
"""

PositionT = _np.ndarray
CoordT = _np.ndarray


@dataclass(frozen=True)
class Translation(Element):
    perms: Tuple[Tuple[int]]
    shift: Tuple[int]

    def __call__(self, sites):
        for i, dim in enumerate(self.shift):
            perm = self.perms[i]
            for j in range(dim):
                sites = _np.take(sites, perm)

        return sites

    def __repr__(self):
        return f"T{self.shift}"


@dataclass(frozen=True)
class PlanarRotation(Element):
    perm: Tuple[int]
    num_rots: int

    def __call__(self, sites):
        for i in range(self.num_rots):
            sites = _np.take(sites, self.perm)

        return sites

    def __repr__(self):
        return f"Rot{self.num_rots}"


@dataclass(frozen=True)
class Reflection(Element):

    perm: Tuple[int]

    def __call__(self, sites):
        sites = _np.take(sites, self.perm)

        return sites

    def __repr__(self):
        return f"Ref"


def get_edges(atoms_positions, cutoff, distance_atol=cutoff_tol):
    cutoff = cutoff + distance_atol
    kdtree = cKDTree(atoms_positions)
    dist_matrix = kdtree.sparse_distance_matrix(kdtree, cutoff)
    id1, id2, values = find(triu(dist_matrix))
    pairs = []
    min_dists = {}  # keys are nodes, values are min dists
    for node in _np.unique(_np.concatenate((id1, id2))):
        min_dist = _np.min(values[(id1 == node) | (id2 == node)])
        min_dists[node] = min_dist
    for node in _np.unique(id1):
        min_dist = _np.min(values[id1 == node])
        mask = (id1 == node) & (_np.isclose(values, min_dist))
        first = id1[mask]
        second = id2[mask]
        for pair in zip(first, second):
            if _np.isclose(min_dist, min_dists[pair[0]]) and _np.isclose(
                min_dist, min_dists[pair[1]]
            ):
                pairs.append(pair)
    return pairs


@dataclass
class LatticeSite:
    """
    Contains information about a single :class:`~netket.graph.Lattice` site.
    """

    id: int
    """Integer ID of this site"""
    position: PositionT
    """Real-space position of this site"""
    cell_coord: CoordT
    """basis coordinates of this site"""

    def __repr__(self):
        s = ", ".join(map(str, (self.id, self.cell_coord)))
        return f"LatticeSite({s})"


def create_sites(
    basis_vectors, extent, apositions, pbc
) -> Tuple[Tuple[LatticeSite, bool], Dict[HashableArray, int]]:
    shell_vec = _np.zeros(extent.size, dtype=int)
    shift_vec = _np.zeros(extent.size, dtype=int)
    # note: by modifying these, the number of shells can be tuned.
    shell_vec[pbc] = 2
    shift_vec[pbc] = 1
    ranges = tuple([list(range(ex)) for ex in extent + shell_vec])
    sites = []
    cell_coord_to_site = {}
    for s_cell in itertools.product(*ranges):
        s_coord_cell = _np.asarray(s_cell) - shift_vec
        inside = not (_np.any(s_coord_cell < 0) or _np.any(s_coord_cell > (extent - 1)))
        atom_count = len(sites)
        for i, atom_coord in enumerate(apositions):
            s_coord_site = s_coord_cell + atom_coord
            r_coord_site = _np.matmul(basis_vectors.T, s_coord_site)
            cell_coord_site = _np.array((*s_coord_cell, i), dtype=int)
            sites.append(
                (
                    LatticeSite(
                        id=None,  # to be set later, after sorting all sites
                        position=r_coord_site,
                        cell_coord=cell_coord_site,
                    ),
                    inside,
                ),
            )
            cell_coord_to_site[HashableArray(cell_coord_site)] = atom_count + i
    return sites, cell_coord_to_site


def get_true_edges(
    basis_vectors: PositionT,
    sites: Sequence[LatticeSite],
    cell_coord_to_site,
    extent,
    distance_atol=cutoff_tol,
):
    positions = _np.array([p.position for p, _ in sites])
    naive_edges = get_edges(
        positions, _np.linalg.norm(basis_vectors, axis=1).max(), distance_atol
    )
    true_edges = []
    for node1, node2 in naive_edges:
        site1, inside1 = sites[node1]
        site2, inside2 = sites[node2]
        if inside1 and inside2:
            true_edges.append((node1, node2))
        elif inside1 or inside2:
            cell1 = site1.cell_coord
            cell2 = site2.cell_coord
            cell1[:-1] = cell1[:-1] % extent
            cell2[:-1] = cell2[:-1] % extent
            node1 = cell_coord_to_site[HashableArray(cell1)]
            node2 = cell_coord_to_site[HashableArray(cell2)]
            edge = (node1, node2)
            if edge not in true_edges and (node2, node1) not in true_edges:
                true_edges.append(edge)
    return true_edges


def deprecated(alternative):
    def wrapper(fn):
        msg = (
            f"{fn.__name__} is deprecated and may be removed in the future. "
            f"You can use `{alternative}`` instead."
        )
        f = _deprecated(msg)(fn)
        return f

    return wrapper


REPR_TEMPLATE = """Lattice(
    n_nodes={},
    extent={},
    basis_vectors=
        {},
    site_offsets=
        {},
)
"""


class Lattice(NetworkX):
    r"""
    A lattice built by periodic arrangement of a given unit cell.

    The lattice is represented as a Bravais lattice with (:code:`basis_vectors`)
    :math:`\{a_d\}_{d=1}^D` (where :math:`D = \mathtt{ndim}` is the dimension of the lattice)
    and a unit cell consisting of one or more sites,
    The positions of those sites within the unit cell can be specified by the :code:`site_offsets`
    parameter. The :code:`extent` is a array where :code:`extent[d]` specifies the number of
    times each unit cell is translated along direction :math:`d`.
    The full lattice is then generated by placing a site at each of the points

    .. math::

        R_{rq} = \sum_{d=1}^D r_d a_d + b_q \in \mathbb R^D

    where :math:`r_d \in \{1, \ldots, \mathtt{extent}[d]\}` and :math:`b_q = \mathtt{site\_offsets}[q]`.
    We also refer to :math:`q` as the `label` of the site within the unit cell.

    The lattice class supports three ways of addressing a specific lattice site:

    id
        An integer index that is used to identify the site in :code:`self.edges()` and
        also corresponds to the index of the corresponding site in sequences like
        :code:`self.nodes()`, :code:`self.positions` or :code:`self.basis_coords`.

    positions
        Real-space position vector :math:`R_{rq}` as defined above, which is available from
        :func:`~netket.graph.Lattice.positions` and can be resolved into an id via
        :func:`~netket.graph.Lattice.id_from_position`.

    basis coordinates
        where each site is specified by a vector :code:`[r1, ..., rD, q]`
        with :math:`r` being the integer vector of length :code:`ndim` specifying the
        cell position as multiples of the primitive vectors and the site label :math:`q`
        giving the number of the site within the unit cell.
        Basis coordinates are available from :func:`~netket.graph.Lattice.basis_coords` and
        can be resolved into an id via :func:`~netket.graph.Lattice.id_from_basis_coords`.
    """
    # Initialization
    # ------------------------------------------------------------------------
    def __init__(
        self,
        basis_vectors: _np.ndarray,
        extent: _np.ndarray,
        *,
        pbc: Union[bool, Sequence[bool]] = True,
        site_offsets: Optional[_np.ndarray] = None,
        atoms_coord: Optional[_np.ndarray] = None,
        distance_atol: float = 1e-5,
    ):
        """
        Constructs a new ``Lattice`` given its side length and the features of the unit cell.

        Args:
            basis_vectors: The basis vectors of the lattice. Should be an array
                of shape `(ndim, ndim)` where each `row` is a basis vector.
            extent: The number of copies of the unit cell; needs to be an array
                of length `ndim`.
            pbc: If ``True`` then the constructed lattice
                will have periodic boundary conditions, otherwise
                open boundary conditions are imposed (default=`True`).
            site_offsets: The position offsets of sites in the unit cell (one site at the origin by default).
            distance_atol: Distance below which spatial points are considered equal for the purpose
                of identifying nearest neighbors.

        Examples:
            Constructs a Kagome lattice with 3 × 3 unit cells:

            >>> import numpy as np
            >>> from netket.graph import Lattice
            >>> # Hexagonal lattice basis
            >>> sqrt3 = np.sqrt(3.0)
            >>> basis = np.array([
            ...     [1.0, 0.0],
            ...     [0.5, sqrt3 / 2.0],
            ... ])
            >>> # Kagome unit cell
            >>> cell = np.array([
            ...     basis[0] / 2.0,
            ...     basis[1] / 2.0,
            ...     (basis[0]+basis[1])/2.0
            ... ])
            >>> g = Lattice(basis_vectors=basis, site_offsets=cell, extent=[3, 3])
            >>> print(g.n_nodes)
            27
            >>> print(g.basis_coords[:6])
            [[0 0 0]
             [0 0 1]
             [0 0 2]
             [0 1 0]
             [0 1 1]
             [0 1 2]]
             >>> print(g.positions[:6])
             [[0.5        0.        ]
              [0.25       0.4330127 ]
              [0.75       0.4330127 ]
              [1.         0.8660254 ]
              [0.75       1.29903811]
              [1.25       1.29903811]]
        """

        self._basis_vectors = self._clean_basis(basis_vectors)
        self._ndim = self._basis_vectors.shape[1]

        self._site_offsets, site_pos_fractional = self._clean_site_offsets(
            site_offsets,
            atoms_coord,
            self._basis_vectors,
        )
        self._pbc = self._clean_pbc(pbc, self._ndim)

        self._extent = _np.asarray(extent, dtype=int)

        sites, self._basis_coord_to_site = create_sites(
            self._basis_vectors, self._extent, site_pos_fractional, pbc
        )
        edges = get_true_edges(
            self._basis_vectors,
            sites,
            self._basis_coord_to_site,
            self._extent,
            distance_atol,
        )
        graph = _nx.MultiGraph(edges)

        # Rename sites
        old_nodes = sorted(set(node for edge in edges for node in edge))
        self._sites = []
        for i, (site, _) in enumerate(sites[old_node] for old_node in old_nodes):
            site.id = i
            self._sites.append(site)
        new_nodes = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
        graph = _nx.relabel_nodes(graph, new_nodes)
        self._basis_coord_to_site = {
            HashableArray(p.cell_coord): p.id for p in self._sites
        }
        self._positions = _np.array([p.position for p in self._sites])
        self._basis_coords = _np.array([p.cell_coord for p in self._sites])

        # Order node names
        edges = list(graph.edges())
        graph = _nx.MultiGraph()
        graph.add_nodes_from([p.id for p in self._sites])
        graph.add_edges_from(edges)

        lattice_dims = _np.expand_dims(self._extent, 1) * self.basis_vectors
        self._inv_dims = _np.linalg.inv(lattice_dims)
        int_positions = self._to_integer_position(self._positions)
        self._int_position_to_site = {
            HashableArray(pos): index for index, pos in enumerate(int_positions)
        }

        super().__init__(graph)

    @staticmethod
    def _clean_basis(basis_vectors):
        """Check and convert `basis_vectors` init argument."""
        basis_vectors = _np.asarray(basis_vectors)
        if basis_vectors.ndim != 2:
            raise ValueError(
                "'basis_vectors' must have ndim==2 (as array of primtive vectors)"
            )
        if basis_vectors.shape[0] != basis_vectors.shape[1]:
            raise ValueError("The number of primitive vectors must match their length")
        return basis_vectors

    @staticmethod
    def _clean_site_offsets(site_offsets, atoms_coord, basis_vectors):
        """Check and convert `site_offsets` init argument."""
        if atoms_coord is not None and site_offsets is not None:
            raise ValueError(
                "atoms_coord is deprecated and replaced by site_offsets, "
                "so both cannot be specified at the same time."
            )
        if atoms_coord is not None:
            warnings.warn(
                "atoms_coord is deprecated and may be removed in future versions, "
                "please use site_offsets instead",
                FutureWarning,
            )
            site_offsets = atoms_coord

        if site_offsets is None:
            site_offsets = _np.zeros(basis_vectors.shape[0])[None, :]

        site_offsets = _np.asarray(site_offsets)
        site_pos_fractional = _np.asarray(
            [
                _np.matmul(_np.linalg.inv(basis_vectors.T), atom_coord)
                for atom_coord in site_offsets
            ]
        )
        if (
            site_pos_fractional.min() < -cutoff_tol
            or site_pos_fractional.max() > 1 + cutoff_tol
        ):
            raise ValueError(
                "site_offsets positions must be contained inside the primitive cell"
            )
        uniques = _np.unique(site_offsets, axis=0)
        if len(site_offsets) != uniques.shape[0]:
            site_offsets = _np.asarray(uniques)
            warnings.warn(
                "Some atom positions are not unique. Duplicates were dropped, and "
                f"now atom positions are {site_offsets}",
                UserWarning,
            )
        return site_offsets, site_pos_fractional

    @staticmethod
    def _clean_pbc(pbc, ndim):
        """Check and convert `pbc` init argument."""
        if isinstance(pbc, bool):
            return _np.array([pbc] * ndim, dtype=bool)
        elif (
            not isinstance(pbc, Sequence)
            or len(pbc) != ndim
            or not all(isinstance(b, bool) for b in pbc)
        ):
            raise ValueError(
                "pbc must be either a boolean or a sequence of booleans with length equal to "
                "the lattice dimenion"
            )
        else:
            return _np.asarray(pbc, dtype=bool)

    # Properties
    # ------------------------------------------------------------------------
    @property
    def basis_vectors(self):
        """Basis vectors of the lattice"""
        return self._basis_vectors

    @property
    def site_offsets(self):
        """Position offsets of sites in the unit cell"""
        return self._site_offsets

    @property
    def ndim(self):
        """Dimension of the lattice"""
        return self._ndim

    @property
    def pbc(self):
        """
        Array of bools such that `pbc[d]` indicates whether dimension d has
        periodic boundaries.
        """
        return self._pbc

    @property
    def extent(self):
        """
        Extent of the lattice
        """
        return self._extent

    @property
    def sites(self) -> Sequence[LatticeSite]:
        """Sequence of lattice site objects"""
        return self._sites

    @property
    def positions(self) -> PositionT:
        """Real-space positions of all lattice sites"""
        return self._positions

    @property
    def basis_coords(self) -> CoordT:
        """basis coordinates of all lattice sites"""
        return self._basis_coords

    # Site lookup
    # ------------------------------------------------------------------------
    def _to_integer_position(self, positions: PositionT) -> Array:
        frac_positions = _np.matmul(positions, self._inv_dims) % 1
        return _np.around(frac_positions * 10 ** tol_digits).astype(int) % (
            10 ** tol_digits
        )

    @staticmethod
    def _get_id_from_dict(
        dict: Dict[HashableArray, int], key: Array
    ) -> Union[int, Array]:
        if key.ndim == 1:
            return dict.get(HashableArray(key), None)
        elif key.ndim == 2:
            return _np.array([dict.get(HashableArray(k), None) for k in key])
        else:
            raise ValueError("Input needs to be rank 1 or rank 2 array")

    def id_from_position(self, position: PositionT) -> Union[int, Array]:
        """
        Return the id for a site at the given position. When passed a rank-2 array
        where each row is a position, this method returns an array of the corresponding ids.
        In both cases, None is returned for positions that do not correspond to a site.
        """
        int_pos = self._to_integer_position(position)
        return self._get_id_from_dict(self._int_position_to_site, int_pos)

    def id_from_basis_coords(self, basis_coords: CoordT) -> Union[int, Array]:
        """
        Return the id for a site at the given basis coordinates. When passed a rank-2 array
        where each row is a coordinate vector, this method returns an array of the corresponding
        ids. In both cases, None is returned for coords that do not correspond to a site.
        """
        key = _np.asarray(basis_coords)
        return self._get_id_from_dict(self._basis_coord_to_site, key)

    def position_from_basis_coords(self, basis_coords: CoordT) -> PositionT:
        """
        Return the position of the site with given basis coordinates.
        When passed a rank-2 array where each row is a coordinate vector,
        this method returns an array of the corresponding positions.
        Throws a KeyError if no site is found for any of the coordinates.
        """
        ids = self.id_from_basis_coords(basis_coords)
        if _np.any(ids == None):  # pylint: disable=singleton-comparison
            raise KeyError(f"No site found at at least one of {basis_coords}")
        return self.positions[ids]

    # Output and drawing
    # ------------------------------------------------------------------------
    def __repr__(self) -> str:
        return REPR_TEMPLATE.format(
            self.n_nodes,
            self._extent,
            str(self.basis_vectors).replace("\n", "\n" + " " * 8),
            str(self.site_offsets).replace("\n", "\n" + " " * 8),
        )

    def draw(
        self,
        ax=None,
        figsize: Optional[Tuple[Union[int, float]]] = None,
        node_color: str = "#1f78b4",
        node_size: int = 300,
        edge_color: str = "k",
        curvature: float = 0.2,
        font_size: int = 12,
        font_color: str = "k",
    ):
        """
        Draws the ``Lattice`` graph

        Args:
            ax: Matplotlib axis object.
            figsize: (width, height) tuple of the generated figure.
            node_color: String with the colour of the nodes.
            node_size: Area of the nodes (as in matplotlib.pyplot.scatter).
            edge_color: String with the colour of the edges.
            curvature: A Bezier curve is fit, where the "height" of the curve is `curvature`
                times the "length" of the curvature.
            font_size: fontsize of the labels for each node.
            font_color: Colour of the font used to label nodes.

        Returns:
            Matplotlib axis object containing the graph's drawing.
        """
        import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel

        # Check if lattice is 1D or 2D... or notnetketwarnings.py
        if self._ndim == 1:
            positions = _np.pad(self.positions, (0, 1), "constant")
        elif self._ndim == 2:
            positions = self.positions
        else:
            raise ValueError(
                f"Make sure that the graph is 1D or 2D in order to be drawn. Now it is {self._ndim}D"
            )
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # FIXME (future) as of 11Apr2021, networkx can draw curved
        # edges only for directed graphs.
        _nx.draw_networkx_edges(
            self.graph.to_directed(),
            pos=positions,
            edgelist=self.edges(),
            connectionstyle=f"arc3,rad={curvature}",
            ax=ax,
            arrowsize=0.1,
            edge_color=edge_color,
            node_size=node_size,
        )
        _nx.draw_networkx_nodes(
            self.graph, pos=positions, ax=ax, node_color=node_color, node_size=node_size
        )
        _nx.draw_networkx_labels(
            self.graph, pos=positions, ax=ax, font_size=font_size, font_color=font_color
        )
        ax.axis("equal")
        return ax

    # Backwards compatibility
    # ------------------------------------------------------------------------
    @deprecated("basis_coords[site_id, -1]")
    def atom_label(self, site_id: int) -> int:
        """`Deprecated`, please use :code:`basis_coords[site_id, -1]` instead."""
        return self.basis_coords[site_id, -1]

    @deprecated("basis_coords[site_id, :-1]")
    def site_to_vector(self, site_id: int) -> CoordT:
        """`Deprecated`, please use :code:`basis_coords[site_id, :-1]` instead."""
        return self.basis_coords[site_id, :-1]

    @deprecated("positions[site_id]")
    def site_to_coord(self, site_id: int) -> PositionT:
        """`Deprecated`, please use :code:`positions[site_id]` instead."""
        return self.positions[site_id]

    @deprecated("id_from_basis_coords([*vector, 0])")
    def vector_to_site(self, vector: CoordT) -> int:
        """`Deprecated`, please use :code:`id_from_basis_coords([*vector, 0])` instead."""
        # Note: This only gives one site within the unit cell, so that
        # `vector_to_site(site_to_vector(i)) == i` is _not_ true in general,
        # which is consistent with the behavior of the v2 lattice.
        return self.id_from_basis_coords([*vector, 0])

    @deprecated("position_from_basis_coords([*vector, label])")
    def vector_to_coord(self, vector: CoordT, label: int) -> PositionT:
        """`Deprecated`, please use :code:`position_from_basis_coords([*vector, label])` instead."""
        return self.position_from_basis_coords([*vector, label])

    @property
    @deprecated("positions")
    def coordinates(self) -> PositionT:
        """`Deprecated`, please use :code:`positions` instead."""
        return self.positions

    @property
    @deprecated("site_offsets")
    def atoms_coord(self) -> PositionT:
        """`Deprecated`, please use :code:`site_offsets` instead."""
        return self._site_offsets

    # Symmetries
    # ------------------------------------------------------------------------
    def _get_id_or_raise(self, pos: PositionT) -> Union[int, Array]:
        ids = self.id_from_position(pos)
        if _np.any(ids == None):  # pylint: disable=singleton-comparison
            raise KeyError(f"No site found at at least one of {pos}")
        return ids

    def translation_perm(self):
        perms = []
        for vec in self.basis_vectors:
            perm = []
            for position in self._positions:
                position = position.copy() + vec
                perm.append(self._get_id_or_raise(position))

            perms.append(tuple(perm))
        return tuple(perms)

    def rotation_perm(self, period, axes=[0, 1]):
        perm = []
        axes = list(axes)
        angle = 2 * pi / period
        rot_mat = _np.array(
            [[_np.cos(angle), -_np.sin(angle)], [_np.sin(angle), _np.cos(angle)]]
        )

        rpositions = self._positions.copy()
        rpositions[:, axes] = _np.matmul(rpositions[:, axes], rot_mat)

        for position in rpositions:
            try:
                perm.append(self._get_id_or_raise(position))
            except KeyError as e:
                raise ValueError(
                    "Rotation with the specified period and axes does not map lattice to itself"
                ) from e

        return tuple(perm)

    def reflection_perm(self, axis=0):
        perm = []
        rpositions = self._positions.copy()
        rpositions[:, axis] = -1 * rpositions[:, axis]

        for position in rpositions:
            try:
                perm.append(self._get_id_or_raise(position))
            except KeyError as e:
                raise ValueError(
                    "Reflection about specified axis does not map lattice to itself"
                ) from e

        return tuple(perm)

    def planar_rotations(self, period, axes=[0, 1]) -> PermutationGroup:
        """
        Returns PermutationGroup corresponding to rotations about specfied axes with specified period

        Arguments:
            period: Period of the rotations
            axes: Axes that define the plane of the rotation
        """

        perm = self.rotation_perm(period, axes)
        rotations = [PlanarRotation(perm, n) for n in range(1, period)]

        return PermutationGroup([Identity()] + rotations, degree=self.n_nodes)

    def basis_translations(self) -> PermutationGroup:
        """
        Returns PermutationGroup corresponding to translations by site_offsets vectors
        """

        translations = product(*[range(i) for i in self._extent])
        next(translations)

        perms = self.translation_perm()
        translations = [Translation(perms, i) for i in translations]

        return PermutationGroup([Identity()] + translations, degree=self.n_nodes)

    def reflections(self, axis=0) -> PermutationGroup:
        """
        Returns PermutationGroup corresponding to reflection about axis
        args:
          axis: Generated reflections about specified axis
        """
        perm = self.reflection_perm(axis)

        return PermutationGroup([Identity()] + [Reflection(perm)], degree=self.n_nodes)
