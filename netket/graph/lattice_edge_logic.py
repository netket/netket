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


import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import find, triu
from netket.utils.float import comparable, is_approx_int

# Near-neighbour search logic


def create_padded_sites(basis_vectors, extent, site_offsets, pbc, order):
    """Generates all lattice sites in an extended shell"""
    # note: by modifying these, the number of shells can be tuned.
    shell_vec = np.where(pbc, 2 * order, 0)
    shift_vec = np.where(pbc, order, 0)

    shell_min = 0 - shift_vec
    shell_max = np.asarray(extent) + shell_vec - shift_vec

    # cell coordinates
    ranges = [slice(lo, hi) for lo, hi in zip(shell_min, shell_max)]
    # site coordinate within unit cell
    ranges += [slice(0, len(site_offsets))]

    basis_coords = np.mgrid[ranges].reshape(len(extent) + 1, -1).T
    positions = basis_coords[:, :-1] @ basis_vectors
    positions = positions.reshape(-1, len(site_offsets), len(extent)) + site_offsets
    positions = positions.reshape(-1, len(extent))

    basis_coords[:, :-1] %= extent
    radix = np.cumprod([1, len(site_offsets), *extent[:0:-1]])[::-1]
    equivalent_ids = basis_coords @ radix

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


def get_custom_edges(basis_vectors, extent, site_offsets, pbc, atol, descriptor):
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
        radix = np.cumprod([len(site_offsets), *extent[:0:-1]])[::-1]
        start = start @ radix + sl1
        end = end @ radix + sl2

        return [(*edge, color) for edge in zip(start, end)]

    colored_edges = []
    for i, desc in enumerate(descriptor):
        if len(desc) == 4:
            colored_edges += translated_edges(*desc)
        elif len(desc) == 3:
            colored_edges += translated_edges(*desc, i)
        else:
            raise ValueError(
                "Each descriptor line is required to contain two sublattice indices, a distance vector, and an optional color"
            )
    return colored_edges
