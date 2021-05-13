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

from .graph import NetworkX
from scipy.spatial import cKDTree
from scipy.sparse import find, triu
import numpy as _np
import itertools
import networkx as _nx
import warnings
from typing import Tuple, Union, Optional
from math import pi
from itertools import product
from dataclasses import dataclass

from netket.utils.semigroup import Identity, Element
from .symmetry import SymmGroup

tol_digits = 5
cutoff_tol = _np.power(10.0, -tol_digits)
"""Tolerance for the maximum distance cutoff when computing the sparse distance matrix.
This is necessary because of floating-point errors when computing the distance in non-trivial 
lattices.
"""


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


def create_points(basis_vectors, extent, atom_coords, pbc):
    shell_vec = _np.zeros(extent.size, dtype=int)
    shift_vec = _np.zeros(extent.size, dtype=int)
    # note: by modifying these, the number of shells can be tuned.
    shell_vec[pbc] = 2
    shift_vec[pbc] = 1
    ranges = tuple([list(range(ex)) for ex in extent + shell_vec])
    atoms = []
    cellANDlabel_to_site = {}
    for s_cell in itertools.product(*ranges):
        s_coord_cell = _np.asarray(s_cell) - shift_vec
        if _np.any(s_coord_cell < 0) or _np.any(s_coord_cell > (extent - 1)):
            inside = False
        else:
            inside = True
        atom_count = len(atoms)
        for i, atom_coord in enumerate(atom_coords):
            s_coord_atom = s_coord_cell + atom_coord
            r_coord_atom = _np.matmul(basis_vectors.T, s_coord_atom)
            atoms.append(
                {
                    "Label": i,
                    "cell": s_coord_cell,
                    "r_coord": r_coord_atom,
                    "inside": inside,
                }
            )
            if tuple(s_coord_cell) not in cellANDlabel_to_site.keys():
                cellANDlabel_to_site[tuple(s_coord_cell)] = {}
            cellANDlabel_to_site[tuple(s_coord_cell)][i] = atom_count + i
    return atoms, cellANDlabel_to_site


def get_true_edges(
    basis_vectors, atoms, cellANDlabel_to_site, extent, distance_atol=cutoff_tol
):
    atoms_positions = dicts_to_array(atoms, "r_coord")
    naive_edges = get_edges(
        atoms_positions, _np.linalg.norm(basis_vectors, axis=1).max(), distance_atol
    )
    true_edges = []
    for node1, node2 in naive_edges:
        atom1 = atoms[node1]
        atom2 = atoms[node2]
        if atom1["inside"] and atom2["inside"]:
            true_edges.append((node1, node2))
        elif atom1["inside"] or atom2["inside"]:
            cell1 = atom1["cell"] % extent
            cell2 = atom2["cell"] % extent
            node1 = cellANDlabel_to_site[tuple(cell1)][atom1["Label"]]
            node2 = cellANDlabel_to_site[tuple(cell2)][atom2["Label"]]
            edge = (node1, node2)
            if edge not in true_edges and (node2, node1) not in true_edges:
                true_edges.append(edge)
    return true_edges


def dicts_to_array(dicts, key):
    result = []
    for d in dicts:
        result.append(d[key])
    return _np.asarray(result)


class Lattice(NetworkX):
    """A lattice built translating a unit cell and adding edges between nearest neighbours sites.

    The unit cell is defined by the ``basis_vectors`` and it can contain an arbitrary number of atoms.
    Each atom is located at an arbitrary position and is labelled by an integer number,
    meant to distinguish between the different atoms within the unit cell.
    Periodic boundary conditions can also be imposed along the desired directions.
    There are three different ways to refer to the lattice sites. A site can be labelled
    by a simple integer number (the site index) or by its coordinates (actual position in space).
    """

    def __init__(
        self,
        basis_vectors,
        extent,
        *,
        pbc: bool = True,
        atoms_coord=[],
        distance_atol: float = 1e-5,
    ):
        """
        Constructs a new ``Lattice`` given its side length and the features of the unit cell.

        Args:
            basis_vectors: The basis vectors of the unit cell.
            extent: The number of copies of the unit cell.
            pbc: If ``True`` then the constructed lattice
                will have periodic boundary conditions, otherwise
                open boundary conditions are imposed (default=``True``).
            atoms_coord: The coordinates of different atoms in the unit cell (default=one atom at the origin).
            distance_atol: A KDTree algorithm finds first neighbours of the lattice, which define the edges of
                the graph. The algorithm needs to be specified some absolute tolerance for those distances,
                as sometimes floating point errors might cause some edges not to be detected.

        Examples:
            Constructs a rectangular 3X4 lattice with periodic boundary conditions.

            >>> import netket
            >>> g=netket.graph.Lattice(basis_vectors=[[1,0],[0,1]],extent=[3,4])
            >>> print(g.n_nodes)
            12

        """

        self._basis_vectors = _np.asarray(basis_vectors)
        if self._basis_vectors.ndim != 2:
            raise ValueError("Every vector must have the same dimension.")
        if self._basis_vectors.shape[0] != self._basis_vectors.shape[1]:
            raise ValueError(
                "basis_vectors must be a basis for the N-dimensional vector space you chose"
            )

        if not atoms_coord:
            atoms_coord = [_np.zeros(self._basis_vectors.shape[0])]
        atoms_coord = _np.asarray(atoms_coord)
        atoms_coord_fractional = _np.asarray(
            [
                _np.matmul(_np.linalg.inv(self._basis_vectors.T), atom_coord)
                for atom_coord in atoms_coord
            ]
        )
        if (
            atoms_coord_fractional.min() < -cutoff_tol
            or atoms_coord_fractional.max() > 1 + cutoff_tol
        ):
            # Maybe there is another way to state this. I want to avoid that there exists the possibility that two atoms from different cells are at the same position:
            raise ValueError(
                "atoms must reside inside their corresponding unit cell, which includes only the 0-faces in fractional coordinates."
            )
        uniques = _np.unique(atoms_coord, axis=0)
        if len(atoms_coord) != uniques.shape[0]:
            atoms_coord = _np.asarray(uniques)
            warnings.warn(
                f"Some atom positions are not unique. Duplicates were dropped, and now atom positions are {atoms_coord}",
                UserWarning,
            )

        self._atoms_coord = atoms_coord

        if isinstance(pbc, bool):
            self._pbc = [pbc] * self._basis_vectors.shape[1]
        elif (
            not isinstance(pbc, list)
            or len(pbc) != self._basis_vectors.shape[1]
            or sum([1 for pbci in pbc if isinstance(pbci, bool)])
            != self._basis_vectors.shape[1]
        ):
            raise ValueError(
                "pbc must be either a boolean or a list of booleans with the same dimension as the vector space you chose."
            )
        else:
            self._pbc = pbc

        extent = _np.asarray(extent)
        self.extent = extent

        atoms, cellANDlabel_to_site = create_points(
            self._basis_vectors, extent, atoms_coord_fractional, pbc
        )
        edges = get_true_edges(
            self._basis_vectors, atoms, cellANDlabel_to_site, extent, distance_atol
        )
        graph = _nx.MultiGraph(edges)

        # Rename atoms
        old_nodes = sorted(set([node for edge in edges for node in edge]))
        self._atoms = [atoms[old_node] for old_node in old_nodes]
        self._coord_to_site = {
            tuple(atom["r_coord"]): new_site
            for new_site, atom in enumerate(self._atoms)
        }
        new_nodes = {old_node: new_node for new_node, old_node in enumerate(old_nodes)}
        graph = _nx.relabel_nodes(graph, new_nodes)

        self._coords = dicts_to_array(self._atoms, "r_coord")
        self._lattice_dims = _np.expand_dims(self.extent, 1) * self.basis_vectors

        # Order node names
        nodes = sorted(graph.nodes())
        edges = list(graph.edges())
        graph = _nx.MultiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

        self._inv_dims = _np.linalg.inv(self._lattice_dims)
        frac_positions = _np.matmul(self._coords, self._inv_dims) % 1
        int_positions = _np.around(frac_positions * 10**tol_digits).astype(int)  %  (10**tol_digits)
        self._hash_positions = {
            hash(element.tobytes()): index
            for index, element in enumerate(int_positions)
        }

        super().__init__(graph)

    @property
    def basis_vectors(self):
        return self._basis_vectors

    @property
    def atoms_coord(self):
        """
        Coordinates of atoms in the unit cell.
        """
        return self._atoms_coord

    @property
    def coords(self):
        """
        Returns list of coordinates of lattice points
        """

        return self._coords

    def translation_perm(self):
        perms = []
        for vec in self.basis_vectors:
            perm = []
            for coord in self._coords:
                hash_coord = coord.copy() + vec
                hash_coord = _np.matmul(hash_coord, self._inv_dims) % 1
                # make sure 1 and 0 are treated the same
                hash_coord = _np.around(hash_coord * 10**tol_digits).astype(int)  %  (10**tol_digits)
                hash_coord = hash(hash_coord.tobytes())
                perm.append(self._hash_positions[hash_coord])

            perms.append(tuple(perm))
        return tuple(perms)

    def rotation_perm(self, period, axes=[0, 1]):
        perm = []
        axes = list(axes)
        angle = 2*pi/period
        rot_mat = _np.array([[_np.cos(angle), -_np.sin(angle)], [_np.sin(angle), _np.cos(angle)]])

        rot_coords = self._coords.copy()
        rot_coords[:, axes] = _np.matmul(rot_coords[:, axes], rot_mat)

        for hash_coord in rot_coords:
            hash_coord = _np.matmul(hash_coord, self._inv_dims) % 1
            hash_coord = _np.around(frac_positions * 10**tol_digits).astype(int) % (10**tol_digits)
            hash_coord = hash(hash_coord.tobytes())
            if hash_coord in self._hash_positions:
                perm.append(self._hash_positions[hash_coord])
            else:
                raise ValueError(
                    "Rotation with the specified period and axes does not map lattice to itself"
                )

        return tuple(perm)

    def reflection_perm(self, axis=0):
        perm = []
        ref_coords = self._coords.copy()
        ref_coords[:, axis] = -1 * ref_coords[:, axis]

        for hash_coord in ref_coords:
            hash_coord = _np.matmul(hash_coord, self._inv_dims) % 1
            hash_coord = _np.around(frac_positions * 10**tol_digits)).astype(int)  %  (10**tol_digits)
            hash_coord = hash(hash_coord.tobytes())
            if hash_coord in self._hash_positions:
                perm.append(self._hash_positions[hash_coord])
            else:
                raise ValueError(
                    "Reflection about specified axis does not map lattice to itself"
                )

        return tuple(perm)

    def planar_rotations(self, period, axes=[0, 1]) -> SymmGroup:
        """
        Returns SymmGroup corresponding to rotations about specfied axes with specified period

        Arguments:
            period: Period of the rotations
            axes: Axes that define the plane of the rotation
        """

        perm = self.rotation_perm(period, axes)
        rotations = [PlanarRotation(perm, n) for n in range(1, period)]

        return SymmGroup([Identity()] + rotations, graph=self)

    def basis_translations(self) -> SymmGroup:
        """
        Returns SymmGroup corresponding to translations by basis vectors
        """

        translations = product(*[range(i) for i in self.extent])
        next(translations)

        perms = self.translation_perm()
        translations = [Translation(perms, i) for i in translations]

        return SymmGroup([Identity()] + translations, graph=self)

    def reflections(self, axis=0) -> SymmGroup:
        """
        Returns SymmGroup corresponding to reflection about axis
        args:
          axis: Generated reflections about specified axis
        """
        perm = self.reflection_perm(axis)

        return SymmGroup([Identity()] + [Reflection(perm)], graph=self)

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
        import matplotlib.pyplot as plt

        # Check if lattice is 1D or 2D... or not
        ndim = len(self._atoms[0]["r_coord"])
        if ndim == 1:
            positions = {
                n: _np.pad(self.site_to_coord(n), (0, 1), "constant")
                for n in self.nodes()
            }
        elif ndim == 2:
            positions = {n: self.site_to_coord(n) for n in self.nodes()}
        else:
            raise ValueError(
                f"Make sure that the graph is 1D or 2D in order to be drawn. Now it is {ndim}D"
            )
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

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

    def atom_label(self, site):
        return self._atoms[site]["Label"]

    def site_to_coord(self, site):
        return self._atoms[site]["r_coord"]

    def coord_to_site(self, coord):
        return self._coord_to_site[tuple(coord)]

    def site_to_vector(self, site):
        return self._atoms[site]["cell"]

    def vector_to_coord(self, vector):
        return _np.matmul(self._basis_vectors, vector)

    def __repr__(self):
        return "Lattice(n_nodes={})\n  extent={}\n  basis_vectors={}".format(
            self.n_nodes, self.extent.tolist(), self.basis_vectors.tolist()
        )
