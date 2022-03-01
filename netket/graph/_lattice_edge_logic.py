# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Tuple, Sequence, Union
from textwrap import dedent

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import find, triu
from netket.utils.float import comparable, is_approx_int
from netket.utils.types import Array


def create_site_positions(basis_vectors, extent, site_offsets, extra_shells=None):
    """Generates the coordinates of all lattice sites.

    extra_shells: (optional) the number of unit cells added along each lattice direction.
        This is used for near-neighbour searching in periodic BCs.
        If used, it must be a vector of the same length as extent"""
    if extra_shells is None:
        extra_shells = np.zeros(extent.size, dtype=int)

    shell_min = -extra_shells
    shell_max = extent + extra_shells

    # cell coordinates
    ranges = [slice(lo, hi) for lo, hi in zip(shell_min, shell_max)]
    # site coordinate within unit cell
    ranges += [slice(0, len(site_offsets))]

    basis_coords = np.mgrid[ranges].reshape(len(extent) + 1, -1).T
    positions = basis_coords[:, :-1] @ basis_vectors
    positions = positions.reshape(-1, len(site_offsets), len(extent)) + site_offsets
    positions = positions.reshape(-1, len(extent))

    return basis_coords, positions


def site_to_idx(basis_coords, extent, site_offsets):
    """Converts unit cell + sublattice coordinates into lattice site indices.

    basis_coords accepted as an array including sublattice coordinate in last column
    or as a tuple of unit cell coordinates and a shared sublattice index
    """
    if isinstance(basis_coords, Tuple):
        basis_coords, sl = basis_coords
    else:
        basis_coords, sl = basis_coords[:, :-1], basis_coords[:, -1]

    # Accepts extended shells
    basis_coords = basis_coords % extent

    # Index difference between sites one lattice site apart in each direction
    # len(site_offsets) for the last axis, as all sites in one cell are listed
    # factor of extent[-1] to the penultimate axis, etc.
    radix = np.cumprod([len(site_offsets), *extent[:0:-1]])[::-1]

    return basis_coords @ radix + sl


# Near-neighbour search logic


def create_padded_sites(basis_vectors, extent, site_offsets, pbc, order):
    """Generates all lattice sites in an extended shell"""
    extra_shells = np.where(pbc, order, 0)
    basis_coords, positions = create_site_positions(
        basis_vectors, extent, site_offsets, extra_shells
    )
    equivalent_ids = site_to_idx(basis_coords, extent, site_offsets)

    return positions, equivalent_ids


def get_naive_edges(positions, cutoff, order):
    """
    Given an array of spatial `positions`, returns a list `es`, so that
    `es[k]` contains all pairs of (k + 1)-nearest neighbors up to `order`.
    Only edges up to distance `cutoff` are considered.
    """
    kdtree = cKDTree(positions)
    dist_matrix = kdtree.sparse_distance_matrix(kdtree, cutoff)
    row, col, dst = find(triu(dist_matrix))
    dst = comparable(dst)
    _, ii = np.unique(dst, return_inverse=True)

    return [sorted(list(zip(row[ii == k], col[ii == k]))) for k in range(order)]


def get_nn_edges(
    basis_vectors,
    extent,
    site_offsets,
    pbc,
    distance_atol,
    order,
):
    """For :code:`order == k`, generates all edges between up to :math:`k`-nearest
    neighbor sites (measured by their Euclidean distance). Edges are colored by length
    with colors between 0 and `order - 1` in order of increasing length."""
    positions, ids = create_padded_sites(
        basis_vectors, extent, site_offsets, pbc, order
    )
    naive_edges_by_order = get_naive_edges(
        positions,
        order * np.linalg.norm(basis_vectors, axis=1).max() + distance_atol,
        order,
    )
    colored_edges = []
    for k, naive_edges in enumerate(naive_edges_by_order):
        true_edges = set()
        for node1, node2 in naive_edges:
            # switch to real node indices
            node1 = ids[node1]
            node2 = ids[node2]
            if node1 == node2:
                raise RuntimeError(
                    f"Lattice contains self-referential edge {(node1, node2)} of order {k}"
                )
            elif node1 > node2:
                node1, node2 = node2, node1
            true_edges.add((node1, node2))
        for edge in true_edges:
            colored_edges.append((*edge, k))
    return colored_edges


# Unit cell distribution logic

CustomEdgeT = Union[Tuple[int, int, Array], Tuple[int, int, Array, int]]


def get_custom_edges(
    basis_vectors, extent, site_offsets, pbc, atol, custom_edges: Sequence[CustomEdgeT]
):
    """Generates the edges described in `custom_edges` for all unit cells.

    See the docstring of `Lattice.__init__` for the syntax of `custom_edges."""
    if not all([len(desc) in (3, 4) for desc in custom_edges]):
        raise ValueError(
            dedent(
                """
            custom_edges must be a list of tuples of length 3 or 4.
            Every tuple must contain two sublattice indices (integers), a distance vector
            and can optionally include an integer to represent the color of that edge.

            Check the docstring of `nk.graph.Lattice` for more informations.
            """
            )
        )

    def translated_edges(sl1, sl2, distance, color):
        # get distance in terms of unit cells
        d_cell = (distance + site_offsets[sl1] - site_offsets[sl2]) @ np.linalg.inv(
            basis_vectors
        )

        if not np.all(is_approx_int(d_cell, atol=atol)):
            # error out
            msg = f"{distance} is invalid distance vector between sublattices {sl1}->{sl2}"
            # see if the user flipped the vector accidentally
            d_cell = (distance + site_offsets[sl2] - site_offsets[sl1]) @ np.linalg.inv(
                basis_vectors
            )
            if np.all(is_approx_int(d_cell, atol=atol)):
                msg += f" (but valid {sl2}->{sl1})"
            raise ValueError(msg)

        d_cell = np.asarray(np.rint(d_cell), dtype=int)

        # catches self-referential and other unrealisably long edges
        if not np.all(d_cell < extent):
            raise ValueError(
                f"Distance vector {distance} does not fit into the lattice"
            )

        # Unit cells of starting points
        start_min = np.where(pbc, 0, np.maximum(0, -d_cell))
        start_max = np.where(pbc, extent, extent - np.maximum(0, d_cell))
        start_ranges = [slice(lo, hi) for lo, hi in zip(start_min, start_max)]
        start = np.mgrid[start_ranges].reshape(len(extent), -1).T
        end = (start + d_cell) % extent

        # Convert to site indices
        start = site_to_idx((start, sl1), extent, site_offsets)
        end = site_to_idx((end, sl2), extent, site_offsets)

        return [(*edge, color) for edge in zip(start, end)]

    colored_edges = []
    for i, desc in enumerate(custom_edges):
        edge_data = desc[:3]
        edge_color = desc[3] if len(desc) == 4 else i
        colored_edges += translated_edges(*edge_data, edge_color)
    return colored_edges
