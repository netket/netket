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

cutoff_tol = 1e-5
"""Tolerance for the maximum distance cutoff when computing the sparse distance matrix.
This is necessary because of floating-point errors when computing the distance in non-trivial 
lattices.
"""


def get_edges(atoms_positions, cutoff):
    cutoff = cutoff + cutoff_tol
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


def get_true_edges(basis_vectors, atoms, cellANDlabel_to_site, extent):
    atoms_positions = dicts_to_array(atoms, "r_coord")
    naive_edges = get_edges(
        atoms_positions, _np.linalg.norm(basis_vectors, axis=1).max()
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

    def __init__(self, basis_vectors, extent, *, pbc: bool = True, atoms_coord=[]):
        """
        Constructs a new ``Lattice`` given its side length and the features of the unit cell.

        Args:
            basis_vectors: The basis vectors of the unit cell.
            extent: The number of copies of the unit cell.
            pbc: If ``True`` then the constructed lattice
                will have periodic boundary conditions, otherwise
                open boundary conditions are imposed (default=``True``).
            atoms_coord: The coordinates of different atoms in the unit cell (default=one atom at the origin).

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
        if atoms_coord_fractional.min() < 0 or atoms_coord_fractional.max() >= 1:
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
        edges = get_true_edges(self._basis_vectors, atoms, cellANDlabel_to_site, extent)
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

        # Order node names
        nodes = sorted(graph.nodes())
        edges = list(graph.edges())
        graph = _nx.MultiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

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
