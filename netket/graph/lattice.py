from graph import NetworkX
from scipy.spatial import cKDTree
from scipy.sparse import find, triu
import numpy as _np
import itertools
import networkx as _nx


class Lattice(NetworkX):
    """ An orthorhombic lattice built translating a unit cell and adding edges between nearest neighbours sites.
        The unit cell is defined by the ``basis_vectors`` and it can contain an arbitrary number of atoms.
        Each atom is located at an arbitrary position and is labelled by an integer number,
        meant to distinguish between the different atoms within the unit cell.
        Periodic boundary conditions can also be imposed along the desired directions.
        There are three different ways to refer to the lattice sites. A site can be labelled
        by a simple integer number (the site index) or by its coordinates (actual position in space)."""

    def __init__(self, basis_vectors, extent, pbc=True, atoms_coord=[]):
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
                "basis_vectors must be an orthogonal basis for the N-dimensional vector space you chose"
            )

        if not atoms_coord:
            atoms_coord = [_np.zeros(self._basis_vectors.shape[0]).tolist()]
        atoms_coord = _np.asarray(atoms_coord)
        if atoms_coord.min() < 0 or atoms_coord.max() >= 1:
            # Maybe there is another way to state this. I want to avoid that there exists the possibility that two atoms from different cells are at the same position:
            raise ValueError(
                "atoms must reside inside their corresponding unit cell, which includes only the 0-faces."
            )
        tuple_array = [tuple(row) for row in atoms_coord]
        uniques = _np.unique(tuple_array)
        if len(atoms_coord) != uniques.shape[0]:
            atoms_coord = uniques
            print(
                "Warning: Some atom positions are not unique. Duplicates were dropped, and now atom positions are {0}".format(
                    atoms_coord
                )
            )

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

        ranges = tuple([list(range(ex)) for ex in extent])
        n_atoms = len(atoms_coord)

        self._coord_to_site = []
        self._site_to_coord = {}
        self._atom_label = []

        for r in itertools.product(*ranges):
            cell_coord = _np.matmul(basis_vectors, r)

            for atom in atoms_coord:
                coord = _np.asarray(atom, dtype=_np.float32) + cell_coord
                self._site_to_coord[len(self._coord_to_site)] = tuple(coord)
                self._coord_to_site.append(tuple(coord))

            self._atom_label += range(n_atoms)

        self._extent = extent
        edges = self.get_edges()
        graph = _nx.MultiGraph(edges)
        super().__init__(graph)

    def get_edges(self):
        atoms_position = _np.asarray(self._coord_to_site)
        boxsize = _np.matmul(self._basis_vectors, self._extent)
        for i, pbci in enumerate(self._pbc):
            if not pbci:
                boxsize[i] = False
        kdtree = cKDTree(atoms_position, boxsize=boxsize)
        dist_matrix = kdtree.sparse_distance_matrix(kdtree, self._basis_vectors.max())
        id1, id2, values = find(
            triu(dist_matrix)
        )  # find non-zero entries of dist_matrix
        pairs = []
        for node in _np.unique(id1):
            min_dist = _np.min(values[id1 == node])
            first = id1[(id1 == node) & (values == min_dist)]
            second = id2[(id1 == node) & (values == min_dist)]
            pairs += list(zip(first, second))
        return pairs

    def atom_label(self, site):
        return self._atom_label[site]

    def site_to_coord(self, site):
        return self._site_to_coord[site]

    def coord_to_site(self, coord):
        return self._coord_to_site.index(tuple(coord))
