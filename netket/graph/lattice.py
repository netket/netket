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
from math import pi

from netket.utils.types import Array
from typing import Callable, Dict, Sequence, Tuple, Union, Optional, TYPE_CHECKING
import warnings

import igraph
import numpy as _np
from scipy.spatial import cKDTree
from scipy.sparse import find, triu

from netket.utils.deprecation import deprecated as _deprecated
from netket.utils import HashableArray
from netket.utils.float import comparable, comparable_periodic, is_approx_int
from netket.utils.group import PointGroup, PermutationGroup, trivial_point_group

from .graph import Graph

if TYPE_CHECKING:
    from .space_group import SpaceGroupBuilder

PositionT = _np.ndarray
CoordT = _np.ndarray


class InvalidSiteError(Exception):
    pass


class InvalidWaveVectorError(Exception):
    pass


@dataclass
class LatticeSite:
    """
    Contains information about a single :class:`~netket.graph.Lattice` site.
    """

    id: int
    """Integer ID of this site"""
    position: PositionT
    """Real-space position of this site"""
    basis_coord: CoordT
    """basis coordinates of this site"""

    def __repr__(self):
        s = ", ".join(map(str, (self.id, self.basis_coord)))
        return f"LatticeSite({s})"


def create_sites(
    basis_vectors, extent, apositions, pbc, order
) -> Tuple[Sequence[LatticeSite], Sequence[bool], Dict[HashableArray, int]]:
    # note: by modifying these, the number of shells can be tuned.
    shell_vec = _np.where(pbc, 2 * order, 0)
    shift_vec = _np.where(pbc, order, 0)

    shell_min = 0 - shift_vec
    shell_max = _np.asarray(extent) + shell_vec - shift_vec
    # cell coordinates
    ranges = [slice(lo, hi) for lo, hi in zip(shell_min, shell_max)]
    # site coordinate within unit cell
    ranges += [slice(0, len(apositions))]

    basis_coords = _np.vstack([_np.ravel(x) for x in _np.mgrid[ranges]]).T
    site_coords = (
        basis_coords[:, :-1]
        + _np.tile(apositions.T, reps=len(basis_coords) // len(apositions)).T
    )
    positions = site_coords @ basis_vectors

    sites = []
    coord_to_site = {}
    for idx, (coord, pos) in enumerate(zip(basis_coords, positions)):
        sites.append(
            LatticeSite(
                id=None,  # to be set later, after sorting all sites
                basis_coord=coord,
                position=pos,
            ),
        )
        coord_to_site[HashableArray(coord)] = idx
    is_inside = ~(
        _np.any(basis_coords[:, :-1] < 0, axis=1)
        | _np.any(basis_coords[:, :-1] > (extent - 1), axis=1)
    )
    return sites, is_inside, coord_to_site


def get_edges(positions, cutoff, order):
    """
    Given an array of spatial `positions`, returns a list `es`, so that
    `es[k]` contains all pairs of (k + 1)-nearest neighbors up to `order`.
    Only edges up to distance `cutoff` are considered.
    """
    kdtree = cKDTree(positions)
    dist_matrix = kdtree.sparse_distance_matrix(kdtree, cutoff)
    row, col, dst = find(triu(dist_matrix))
    dst = comparable(dst)
    _, ii = _np.unique(dst, return_inverse=True)

    return [sorted(list(zip(row[ii == k], col[ii == k]))) for k in range(order)]


def get_true_edges(
    basis_vectors: PositionT,
    sites: Sequence[LatticeSite],
    inside: Sequence[bool],
    basis_coord_to_site,
    extent,
    distance_atol,
    order,
):
    positions = _np.array([p.position for p in sites])
    naive_edges_by_order = get_edges(
        positions,
        order * _np.linalg.norm(basis_vectors, axis=1).max() + distance_atol,
        order,
    )
    true_edges_by_order = []
    for k, naive_edges in enumerate(naive_edges_by_order):
        true_edges = set()
        for node1, node2 in naive_edges:
            site1, inside1 = sites[node1], inside[node1]
            site2, inside2 = sites[node2], inside[node2]
            if inside1 and inside2:
                true_edges.add((node1, node2))
            elif inside1 or inside2:
                cell1 = site1.basis_coord
                cell2 = site2.basis_coord
                cell1[:-1] = cell1[:-1] % extent
                cell2[:-1] = cell2[:-1] % extent
                node1 = basis_coord_to_site[HashableArray(cell1)]
                node2 = basis_coord_to_site[HashableArray(cell2)]
                edge = (node1, node2)
                if edge not in true_edges and (node2, node1) not in true_edges:
                    if node1 == node2:
                        raise RuntimeError(
                            f"Lattice contains self-referential edge {(node1, node2)} of order {k}"
                        )
                    true_edges.add(edge)
        true_edges_by_order.append(list(true_edges))
    return true_edges_by_order


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


class Lattice(Graph):
    r"""
    A lattice built by periodic arrangement of a given unit cell.

    The lattice is represented as a Bravais lattice with (:code:`basis_vectors`)
    :math:`\{a_d\}_{d=1}^D` (where :math:`D = \mathtt{ndim}` is the dimension of the
    lattice) and a unit cell consisting of one or more sites,
    The positions of those sites within the unit cell can be specified by the
    :code:`site_offsets` parameter. The :code:`extent` is a array where
    :code:`extent[d]` specifies the number of times each unit cell is translated along
    direction :math:`d`.
    The full lattice is then generated by placing a site at each of the points

    .. math::

        R_{rq} = \sum_{d=1}^D r_d a_d + b_q \in \mathbb R^D

    where :math:`r_d \in \{1, \ldots, \mathtt{extent}[d]\}` and
    :math:`b_q = \mathtt{site\_offsets}[q]`.
    We also refer to :math:`q` as the `label` of the site within the unit cell.

    The lattice class supports three ways of addressing a specific lattice site:

    id
        An integer index that is used to identify the site in :code:`self.edges()` and
        also corresponds to the index of the corresponding site in sequences like
        :code:`self.nodes()`, :code:`self.positions` or :code:`self.basis_coords`.

    positions
        Real-space position vector :math:`R_{rq}` as defined above, which is available
        from :func:`~netket.graph.Lattice.positions` and can be resolved into an id via
        :func:`~netket.graph.Lattice.id_from_position`.

    basis coordinates
        where each site is specified by a vector :code:`[r1, ..., rD, q]`
        with :math:`r` being the integer vector of length :code:`ndim` specifying the
        cell position as multiples of the primitive vectors and the site label :math:`q`
        giving the number of the site within the unit cell.
        Basis coordinates are available from :func:`~netket.graph.Lattice.basis_coords`
        and can be resolved into an id via
        :func:`~netket.graph.Lattice.id_from_basis_coords`.
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
        point_group: Optional[PointGroup] = None,
        max_neighbor_order: int = 1,
    ):
        """
        Constructs a new ``Lattice`` given its side length and the features of the unit
        cell.

        Args:
            basis_vectors: The basis vectors of the lattice. Should be an array
                of shape `(ndim, ndim)` where each `row` is a basis vector.
            extent: The number of copies of the unit cell; needs to be an array
                of length `ndim`.
            pbc: If ``True`` then the constructed lattice
                will have periodic boundary conditions, otherwise
                open boundary conditions are imposed. Can also be an boolean sequence
                of length `ndim`, indicating either open or closed boundary conditions
                separately for each direction.
            site_offsets: The position offsets of sites in the unit cell (one site at
                the origin by default).
            distance_atol: Distance below which spatial points are considered equal for
                the purpose of identifying nearest neighbors.
            point_group: Default `PointGroup` object for constructing space groups
            max_neighbor_order: For :code:`max_neighbor_order == k`, edges between up
                to :math:`k`-nearest neighbor sites (measured by their Euclidean distance)
                are included in the graph. The edges can be distiguished by their color,
                which is set to :math:`k - 1` (so nearest-neighbor edges have color 0).

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
        if max_neighbor_order < 1:
            raise ValueError("max_neighbor_order must be >= 1.")

        self._basis_vectors = self._clean_basis(basis_vectors)
        self._ndim = self._basis_vectors.shape[1]

        self._site_offsets, site_pos_fractional = self._clean_site_offsets(
            site_offsets,
            atoms_coord,
            self._basis_vectors,
        )
        self._pbc = self._clean_pbc(pbc, self._ndim)

        self._extent = _np.asarray(extent, dtype=int)

        self._point_group = point_group

        sites, inside, self._basis_coord_to_site = create_sites(
            self._basis_vectors,
            self._extent,
            site_pos_fractional,
            self._pbc,
            max_neighbor_order,
        )
        edges = get_true_edges(
            self._basis_vectors,
            sites,
            inside,
            self._basis_coord_to_site,
            self._extent,
            distance_atol,
            max_neighbor_order,
        )

        old_nodes = sorted(
            set(node for edges_k in edges for edge in edges_k for node in edge)
        )
        new_nodes = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}

        edges_by_order = []
        for edges_k in edges:
            graph = igraph.Graph()
            graph.add_vertices(len(old_nodes))
            graph.add_edges(
                [(new_nodes[edge[0]], new_nodes[edge[1]]) for edge in edges_k]
            )
            graph.simplify()
            edges_by_order.append(list(graph.get_edgelist()))

        self._sites = []
        for i, site in enumerate(sites[old_node] for old_node in old_nodes):
            site.id = i
            self._sites.append(site)
        self._basis_coord_to_site = {
            HashableArray(p.basis_coord): p.id for p in self._sites
        }
        self._positions = _np.array([p.position for p in self._sites])
        self._basis_coords = _np.array([p.basis_coord for p in self._sites])
        self._lattice_dims = _np.expand_dims(self._extent, 1) * self.basis_vectors
        self._inv_dims = _np.linalg.inv(self._lattice_dims)
        int_positions = self._to_integer_position(self._positions)
        self._int_position_to_site = {
            HashableArray(pos): index for index, pos in enumerate(int_positions)
        }

        colored_edges = [(*e, k) for k, es in enumerate(edges_by_order) for e in es]
        super().__init__(colored_edges, len(self._sites))

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
        fractional_coords = site_offsets @ _np.linalg.inv(basis_vectors)
        fractional_coords_int = comparable_periodic(fractional_coords)
        # Check for duplicates (also across unit cells)
        uniques, idx = _np.unique(fractional_coords_int, axis=0, return_index=True)
        if len(site_offsets) != len(uniques):
            site_offsets = site_offsets[idx]
            fractional_coords = fractional_coords[idx]
            fractional_coords_int = fractional_coords_int[idx]
            warnings.warn(
                "Some atom positions are not unique. Duplicates were dropped, and "
                f"now atom positions are {site_offsets}",
                UserWarning,
            )
        # Check if any site is outside primitive cell (may cause KDTree to malfunction)
        if _np.any(fractional_coords_int < comparable(0.0)) or _np.any(
            fractional_coords_int > comparable(1.0)
        ):

            warnings.warn(
                "Some sites were specified outside the primitive unit cell. This may"
                "cause errors in automatic edge finding.",
                UserWarning,
            )
        return site_offsets, fractional_coords

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
                "pbc must be either a boolean or a sequence of booleans with length"
                "equal to  the lattice dimenion"
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
        frac_positions = _np.matmul(positions, self._inv_dims)
        return comparable_periodic(frac_positions, self.pbc)

    @staticmethod
    def _get_id_from_dict(
        dict: Dict[HashableArray, int], key: Array
    ) -> Union[int, Array]:
        try:
            if key.ndim == 1:
                return dict[HashableArray(key)]
            elif key.ndim == 2:
                return _np.array([dict[HashableArray(k)] for k in key])
            else:
                raise ValueError("Input needs to be rank 1 or rank 2 array")
        except KeyError as e:
            raise InvalidSiteError(
                "Some coordinates do not correspond to a valid lattice site"
            ) from e

    def id_from_position(self, position: PositionT) -> Union[int, Array]:
        """
        Returns the id for a site at the given position. When passed a rank-2 array
        where each row is a position, returns an array of the corresponding ids.
        Throws an `InvalidSiteError` if any of the positions do not correspond
        to a site.
        """
        int_pos = self._to_integer_position(position)
        ids = self._get_id_from_dict(self._int_position_to_site, int_pos)
        return ids

    def id_from_basis_coords(self, basis_coords: CoordT) -> Union[int, Array]:
        """
        Return the id for a site at the given basis coordinates. When passed a rank-2
        array where each row is a coordinate vector, returns an array of the
        corresponding ids. Throws an `InvalidSiteError` if any of the coords do
        not correspond to a site.
        """
        key = _np.asarray(basis_coords)
        return self._get_id_from_dict(self._basis_coord_to_site, key)

    def position_from_basis_coords(self, basis_coords: CoordT) -> PositionT:
        """
        Return the position of the site with given basis coordinates.
        When passed a rank-2 array where each row is a coordinate vector,
        this method returns an array of the corresponding positions.
        Throws an `InvalidSiteError` if no site is found for any of the coordinates.
        """
        ids = self.id_from_basis_coords(basis_coords)
        return self.positions[ids]

    def to_reciprocal_lattice(self, ks: Array) -> Array:
        """
        Converts wave vectors from Cartesian axes to reciprocal lattice vectors.

        Arguments:
            ks: wave vectors in Cartesian axes. Multidimensional arrays are accepted,
                the Cartesian coordinates must form the last dimension.

        Returns:
            The same wave vectors in the reciprocal basis **of the simulation box.**
            Valid wave vector components in this basis are integers in (periodic BCs)
            or zero (in open BCs).

        Throws an `InvalidWaveVectorError` if any of the supplied wave vectors
        are not reciprocal lattice vectors of the simulation box.
        """
        # Ensure that ks has at least 2 dimensions
        ks = _np.asarray(ks)
        if ks.ndim == 1:
            ks = ks[_np.newaxis, :]

        result = ks @ self._lattice_dims.T / (2 * pi)
        # Check that these are integers
        is_valid = is_approx_int(result)
        if not _np.all(is_valid):
            raise InvalidWaveVectorError(
                "Some wave vectors are not reciprocal lattice vectors of the simulation"
                "box spanned by\n"
                + "\n".join(
                    [
                        str(self._lattice_dims[i])
                        + (" (PBC)" if self.pbc[i] else " (OBC)")
                        for i in range(self.ndim)
                    ]
                )
            )

        result = _np.asarray(_np.rint(result), dtype=int)
        # For axes with non-periodic BCs, the k-component must be 0
        is_valid = _np.logical_or(self.pbc, result == 0)
        if not _np.all(is_valid):
            raise InvalidWaveVectorError(
                "Some wave vectors are inconisistent with open boundary conditions"
            )

        return result

    # Generating space groups
    # -----------------------------------------------------------------------
    def space_group_builder(
        self, point_group: Optional[PointGroup] = None
    ) -> "SpaceGroupBuilder":
        """
        Returns a `SpaceGroupBuilder` object that represents the spatial symmetries of
        `self`.

        Arguments:
            point_group: a `PointGroup` object describing the point-group
                         symmetries of `self`. Optional, if not supplied, the
                         `PointGroup` object provded at construction is used.

        Returns:
            A `SpaceGroupBuilder` object that generates `PermutationGroup`s
            encoding the action of `point_group`, the translation group of `self`,
            and the space group obtained as their semidirect product as
            permutations of the sites of `self`. It also yields space group irreps
            for symmetrising wave functions.
        """
        from .space_group import SpaceGroupBuilder

        if point_group is None:
            if isinstance(self._point_group, PointGroup):
                point_group = self._point_group
            elif isinstance(self._point_group, Callable):
                self._point_group = self._point_group()
                point_group = self._point_group
            else:
                raise RuntimeError(
                    "space_group_builder() missing required argument 'point_group'\n"
                    "(lattice has no default point group)"
                )
        return SpaceGroupBuilder(self, point_group)

    def space_group(self, point_group: Optional[PointGroup] = None) -> PermutationGroup:
        """
        Returns the space group generated by the translation symmetries of `self`
        and the elements of `point_group` as a `PermutationGroup` acting on the
        sites of `self`.
        If no `point_group` is specified, uses the point group provided upon
        construction.
        """
        return self.space_group_builder(point_group).space_group

    def point_group(self, point_group: Optional[PointGroup] = None) -> PermutationGroup:
        """
        Returns the action of `point_group` on the sites of `self` as a
        `PermutationGroup`. If no `point_group` is specified, uses the point group
        provided upon construction.
        """
        return self.space_group_builder(point_group).point_group

    def rotation_group(
        self, point_group: Optional[PointGroup] = None
    ) -> PermutationGroup:
        """
        Returns the action of rotations (i.e. symmetries with determinant +1) in
        `point_group` on the sites of `self` as a `PermutationGroup`.
        If no `point_group` is specified, uses the point group provided upon
        construction.
        """
        return self.space_group_builder(point_group).rotation_group

    def translation_group(
        self, dim: Optional[Union[int, Sequence[int]]] = None
    ) -> PermutationGroup:
        """
        Returns the group of lattice translations of `self` as a `PermutationGroup`
        acting on the sites of `self`.
        """
        return self.space_group_builder(
            trivial_point_group(self.ndim)
        ).translation_group(dim)

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
            curvature: A Bezier curve is fit, where the "height" of the curve is
                `curvature` times the "length" of the curvature.
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
                "Make sure that the graph is 1D or 2D in order to be drawn. "
                f" Now it is {self._ndim}D"
            )
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        for edge in self.edges():
            x1, y1 = positions[edge[0]]
            x2, y2 = positions[edge[1]]
            annotation = ax.annotate(
                "",
                xy=(x1, y1),
                xycoords="data",
                xytext=(x2, y2),
                textcoords="data",
                arrowprops=dict(
                    arrowstyle="-",
                    color=edge_color,
                    shrinkA=0,
                    shrinkB=0,
                    patchA=None,
                    patchB=None,
                    connectionstyle=f"arc3,rad={curvature}",
                ),
            )
        ax.scatter(
            *positions.T,
            s=node_size,
            c=node_color,
            marker="o",
            zorder=annotation.get_zorder() + 1,
        )
        for node in self.nodes():
            x1, y1 = positions[node]
            ax.text(
                x1,
                y1,
                str(node),
                horizontalalignment="center",
                verticalalignment="center",
                fontsize=font_size,
                color=font_color,
                zorder=annotation.get_zorder() + 1,
            )
        ax.axis("equal")
        return ax

    # Backwards compatibility
    # ------------------------------------------------------------------------
    @deprecated("basis_coords[site_id, -1]")
    def atom_label(self, site_id: int) -> int:
        """Deprecated. please use :code:`basis_coords[site_id, -1]` instead."""
        return self.basis_coords[site_id, -1]

    @deprecated("basis_coords[site_id, :-1]")
    def site_to_vector(self, site_id: int) -> CoordT:
        """Deprecated. please use :code:`basis_coords[site_id, :-1]` instead."""
        return self.basis_coords[site_id, :-1]

    @deprecated("positions[site_id]")
    def site_to_coord(self, site_id: int) -> PositionT:
        """Deprecated. please use :code:`positions[site_id]` instead."""
        return self.positions[site_id]

    @deprecated("id_from_basis_coords([*vector, 0])")
    def vector_to_site(self, vector: CoordT) -> int:
        """Deprecated. please use :code:`id_from_basis_coords([*vector, 0])` instead."""
        # Note: This only gives one site within the unit cell, so that
        # `vector_to_site(site_to_vector(i)) == i` is _not_ true in general,
        # which is consistent with the behavior of the v2 lattice.
        return self.id_from_basis_coords([*vector, 0])

    @deprecated("position_from_basis_coords([*vector, label])")
    def vector_to_coord(self, vector: CoordT, label: int) -> PositionT:
        "Deprecated. please use :code:`position_from_basis_coords([*vector, label])`."
        return self.position_from_basis_coords([*vector, label])

    @property
    @deprecated("positions")
    def coordinates(self) -> PositionT:
        """Deprecated. please use :code:`positions` instead."""
        return self.positions

    @property
    @deprecated("site_offsets")
    def atoms_coord(self) -> PositionT:
        """Deprecated. please use :code:`site_offsets` instead."""
        return self._site_offsets
