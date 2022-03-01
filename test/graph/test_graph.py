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

import pytest

import netket as nk
import networkx as nx
import math
from math import pi
import numpy as np
import igraph as ig

from netket.graph import (
    Graph,
    Hypercube,
    Grid,
    Lattice,
    Edgeless,
    Triangular,
    Honeycomb,
    Kagome,
    KitaevHoneycomb,
)
from netket.graph import _lattice
from netket.utils import group

from .. import common

pytestmark = common.skipif_mpi

graphs = [
    # star and tree
    Graph.from_igraph(ig.Graph.Star(5)),
    Graph.from_igraph(ig.Graph.Tree(n=3, children=2)),
    # Grid graphs
    Hypercube(length=10, n_dim=1, pbc=True),
    Hypercube(length=4, n_dim=2, pbc=True),
    Hypercube(length=5, n_dim=1, pbc=False),
    Grid([2, 2], pbc=False),
    Grid([4, 2], pbc=[True, False]),
    # lattice graphs
    Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[3, 3],
        pbc=[False, False],
        site_offsets=[[0, 0]],
    ),
    Lattice(
        basis_vectors=[[1.5, math.sqrt(3) / 2.0], [0, math.sqrt(3)]],
        extent=[3, 5],
        site_offsets=[[0, 0], [1, 1]],
    ),
    Lattice(
        basis_vectors=[
            [1.0, 0.0, 0.0],
            [1.0 / 2.0, math.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        extent=[2, 3, 4],
        site_offsets=[[0, 0, 0]],
    ),
    Lattice(
        basis_vectors=[[2.0, 0.0], [1.0, math.sqrt(3)]],
        extent=[4, 4],
        site_offsets=[[0, 0], [1.0 / 2.0, math.sqrt(3) / 2.0], [1.0, 0.0]],
    ),
    # edgeless graph
    Edgeless(10),
]

symmetric_graph_names = [
    "square",
    "triangular",
    "honeycomb",
    "kagome",
    "cubic",
    "bcc",
    "fcc",
    "diamond",
    "pyrochlore",
]

symmetric_graphs = [
    # Square
    nk.graph.Square(3),
    # Triangular
    nk.graph.Triangular([3, 3]),
    # Honeycomb
    nk.graph.Honeycomb([3, 3]),
    # Kagome
    nk.graph.Kagome([3, 3]),
    # Kitaev honeycomb
    nk.graph.KitaevHoneycomb([3, 3]),
    # Cube
    nk.graph.Hypercube(length=3, n_dim=3),
    # Body-centred Cubic
    nk.graph.BCC([3, 3, 3]),
    # Face-centred cubic
    nk.graph.FCC([3, 3, 3]),
    # Diamond
    nk.graph.Diamond([3, 3, 3]),
    # Pyrochlore
    nk.graph.Pyrochlore([3, 3, 3]),
]

unit_cells = [9, 9, 9, 9, 9, 27, 27, 27, 27, 27]

atoms_per_unit_cell = [1, 1, 2, 3, 2, 1, 1, 1, 2, 4]

coordination_number = [4, 6, 3, 4, 3, 6, 8, 12, 4, 6]

dimension = [2, 2, 2, 2, 2, 3, 3, 3, 3, 3]

kvec = [(2 * pi / 3, 0)] + [(4 * pi / 3, 0)] * 4 + [(4 * pi / 3, 0, 0)] * 5

little_group_size = [2] + [6] * 3 + [1] + [8] * 5

little_group_irreps = [2] + [3] * 3 + [1] + [5] * 5


def test_next_neighbors():
    graph1 = nk.graph.Honeycomb(extent=[3, 3], max_neighbor_order=2)
    graph2 = nk.graph.Honeycomb(extent=[3, 3], max_neighbor_order=1)
    assert graph1.n_edges == 3 * graph2.n_edges


def test_custom_edges():
    graph = nk.graph.KitaevHoneycomb(extent=[3, 3])
    for i in range(3):
        assert len(graph.edges(filter_color=i)) == 9

    graph = nk.graph.KitaevHoneycomb([3, 3], pbc=False)
    assert len(graph.edges(filter_color=0)) == 9
    assert len(graph.edges(filter_color=1)) == 6
    assert len(graph.edges(filter_color=2)) == 6

    graph = nk.graph.Lattice(
        np.eye(2), (6, 4), pbc=False, custom_edges=[(0, 0, [1, 0]), (0, 0, [0, 1])]
    )
    assert len(graph.edges(filter_color=0)) == 20
    assert len(graph.edges(filter_color=1)) == 18


@pytest.mark.parametrize("i,name", list(enumerate(symmetric_graph_names)))
def test_lattice_graphs(i, name):
    graph = symmetric_graphs[i]
    # Check to see if graphs have the correct number of nodes and edges
    assert graph.n_nodes == unit_cells[i] * atoms_per_unit_cell[i]
    assert graph.n_edges == graph.n_nodes * coordination_number[i] // 2


# netket#743 : multiple edges
def test_no_redundant_edges():
    g = Grid([3, 2])
    print(g.edges())
    assert g.n_nodes == 6
    assert g.n_edges == 9


@pytest.mark.parametrize("g", graphs + symmetric_graphs)
def test_lattice(g):
    if not isinstance(g, Lattice):
        return

    # sites should be sorted in lexicographic order by basis coordinate
    sort = np.lexsort(g.basis_coords.T[::-1])
    print(g.basis_coords[sort])
    np.testing.assert_almost_equal(sort, np.arange(g.n_nodes))

    # check lookup with id
    for i, site_id in enumerate(g.nodes()):
        assert i == site_id
        cc = g.basis_coords[i]
        pos = g.positions[i]
        manual_pos = g.basis_vectors.T @ cc[:-1] + g.site_offsets[cc[-1]]
        np.testing.assert_almost_equal(manual_pos, pos)

        assert g.id_from_position(pos) == i
        assert g.id_from_basis_coords(cc) == i

    # check lookup with arrays
    if g.n_nodes > 1:
        pos = g.positions[[0, 1]]
        ids = g.id_from_position(pos)
        # assert ids.ndim == 1 and ids.size == 2
        np.testing.assert_almost_equal(ids, [0, 1])

        ccs = g.basis_coords[[0, 1]]
        ids = g.id_from_basis_coords(ccs)
        # assert ids.ndim == 1 and ids.size == 2
        np.testing.assert_almost_equal(ids, [0, 1])

        pos2 = g.position_from_basis_coords(ccs)
        np.testing.assert_almost_equal(pos2, pos)


def test_lattice_site_lookup():
    g = Lattice([[1]], [2])
    pos = [[0.0], [1.0]]
    ids = g.id_from_position(pos)
    np.testing.assert_almost_equal(ids, [0, 1])

    with pytest.raises(_lattice.InvalidSiteError):
        g.id_from_position([[0.5]])
    with pytest.raises(_lattice.InvalidSiteError):
        g.id_from_position([0.5])

    with pytest.raises(_lattice.InvalidSiteError):
        pos = g.position_from_basis_coords([2])
    with pytest.raises(_lattice.InvalidSiteError):
        pos = g.position_from_basis_coords([[2]])


def test_lattice_old_interface():
    with pytest.warns(FutureWarning):
        _ = Lattice(basis_vectors=[[1.0]], atoms_coord=[[0.0], [0.5]], extent=[4])

    def check_alternative(method, alternative):
        with pytest.warns(FutureWarning):
            result = method()
        np.testing.assert_almost_equal(alternative(), result)

    for g in graphs + symmetric_graphs:
        if not isinstance(g, Lattice):
            continue
        check_alternative(lambda: g.atom_label(0), lambda: g.basis_coords[0, -1])
        check_alternative(lambda: g.site_to_vector(0), lambda: g.basis_coords[0, :-1])
        check_alternative(lambda: g.site_to_coord(0), lambda: g.positions[0])

        *cell1, label1 = g.basis_coords[1]
        check_alternative(
            lambda: g.vector_to_site(cell1),
            lambda: g.id_from_basis_coords([*cell1, 0]),
        )
        check_alternative(
            lambda: g.vector_to_coord(cell1, label1),
            lambda: g.position_from_basis_coords([*cell1, label1]),
        )

        check_alternative(lambda: g.coordinates, lambda: g.positions)
        check_alternative(lambda: g.atoms_coord, lambda: g.site_offsets)


@pytest.mark.parametrize("i,name", list(enumerate(symmetric_graph_names)))
def test_lattice_symmetry(i, name):
    graph = symmetric_graphs[i]
    # Try an invalid symmetry group and fail
    with pytest.raises(_lattice.InvalidSiteError):
        if dimension[i] == 2:
            _ = graph.space_group(group.planar.C(5))
        else:
            _ = graph.space_group(group.axial.C(5))
    # Generate space group using the preloaded point group
    sgb = graph.space_group_builder()

    if graph._point_group.is_symmorphic:
        # Check if the point permutation group is isomorphic to geometric one
        np.testing.assert_almost_equal(
            sgb.point_group.product_table, graph._point_group.product_table
        )
    else:
        # If non-symmorphic, point permutation group shouldn't close
        with pytest.raises(RuntimeError):
            pt = sgb.point_group.product_table

    # Build translation group product table explicitly and compare
    pt_1d = [[0, 1, 2], [2, 0, 1], [1, 2, 0]]
    ones = np.ones((3, 3), dtype=int)
    pt = np.kron(pt_1d, ones) * 3 + np.kron(ones, pt_1d)
    if dimension[i] == 3:
        pt = np.kron(pt, ones) * 3 + np.kron(np.ones((9, 9), dtype=int), pt_1d)
    np.testing.assert_almost_equal(pt, sgb.translation_group().product_table)

    # ensure that all space group symmetries are unique and automorphisms
    # don't do this for pyrochlore that takes >10x longer than any other one
    if name != "pyrochlore":
        _check_symmgroups(graph)

    # Try an invalid wave vector and fail
    with pytest.raises(_lattice.InvalidWaveVectorError):
        _ = sgb.little_group([1] * dimension[i])

    # The little group of Γ is the full point group
    assert sgb.little_group(np.zeros(dimension[i])) == sgb.point_group_

    # Generate little groups and their irreps
    assert len(sgb.little_group(*kvec[i])) == little_group_size[i]
    assert len(sgb.little_group(kvec[i])) == little_group_size[i]
    irrep_from_lg = sgb.space_group_irreps(kvec[i])
    irrep_from_sg = sgb.space_group.character_table()
    for irrep in irrep_from_lg:
        assert np.any(np.all(np.isclose(irrep, irrep_from_sg), axis=1))


def coord2index(xs, length):
    if isinstance(xs, int):
        return xs
    i = 0
    scale = 1
    for x in xs:
        i += scale * x
        scale *= length
    return i


def test_graph_wrong():
    with pytest.raises(TypeError):
        nk.graph.Graph(5)

    with pytest.raises(ValueError):
        nk.graph.Graph([1, 2, 3], True)

    with pytest.raises(ValueError):
        nk.graph.Graph([1, 2, 3], [1, 2, 3])

    with pytest.raises(TypeError):
        nk.graph.Hypercube([5])

    with pytest.raises(TypeError):
        nk.graph.Cube([5])

    with pytest.raises(TypeError):
        nk.graph.Square([5])

    with pytest.raises(TypeError):
        nk.graph.Chain([5])


def test_edges_are_correct():
    def check_edges(length, n_dim, pbc):
        x = nx.grid_graph(dim=[length] * n_dim, periodic=pbc)
        x_edges = [[coord2index(i, length) for i in edge] for edge in x.edges()]
        x_edges = sorted([sorted(ed) for ed in x_edges])
        y = nk.graph.Hypercube(length=length, n_dim=n_dim, pbc=pbc)
        y_edges = sorted([sorted(ed) for ed in y.edges()])
        assert x_edges == y_edges

    # with pytest.raises(ValueError):
    #    check_edges(1, 1, False)
    #    check_edges(1, 2, False)

    for length in [3, 4]:
        for dim in [1, 2]:
            for pbc in [True, False]:
                check_edges(length, dim, pbc)
    for pbc in [True, False]:
        check_edges(3, 5, pbc)


def test_nodes():
    count = lambda it: sum(1 for _ in it)
    for graph in graphs:
        nodes = graph.nodes()
        assert count(nodes) == graph.n_nodes


def tonx(graph):
    adl = graph.adjacency_list()
    i = 0
    edges = []
    for els in adl:
        for el in els:
            edges.append([i, el])
        i += 1
    if edges:
        return nx.from_edgelist(edges)

    gx = nx.Graph()
    for i in range(graph.n_nodes):
        gx.add_node(i)
    return gx


def test_draw_lattices():
    # Just checking that lattices are drawn:
    lattices = [graph for graph in graphs if isinstance(graph, Lattice)]
    for lattice in lattices:
        ndim = lattice.ndim
        if ndim not in [1, 2]:
            with pytest.raises(ValueError):
                lattice.draw()
        else:
            _ = lattice.draw(
                figsize=(1.2, 3),
                node_color="blue",
                node_size=600,
                edge_color="green",
                curvature=0.5,
                font_size=20,
                font_color="yellow",
            )


def test_size_is_positive():
    for graph in graphs:
        assert graph.n_nodes > 0
        assert graph.n_edges >= 0


def test_is_connected():
    for i in range(5, 10):
        for j in range(i + 1, i * i):
            x = nx.dense_gnm_random_graph(i, j)
            y = Graph.from_networkx(x)

            if len(x) == len(
                set((i for (i, j) in x.edges)) | set((j for (i, j) in x.edges))
            ):
                assert y.is_connected() == nx.is_connected(x)
            else:
                assert not nx.is_connected(x)


def test_is_bipartite():
    for i in range(1, 10):
        for j in range(1, i * i):
            x = nx.dense_gnm_random_graph(i, j)
            y = Graph.from_networkx(x)
            if len(x) == len(
                set((i for (i, j) in x.edges())) | set((j for (i, j) in x.edges()))
            ):
                assert y.is_bipartite() == nx.is_bipartite(x)


def test_computes_distances():
    for graph in graphs:
        if graph.is_connected():
            nxg = nx.from_edgelist(graph.edges())
            d = graph.distances()
            d1 = dict(nx.shortest_path_length(nxg))
            for i in range(graph.n_nodes):
                for j in range(graph.n_nodes):
                    assert d1[i][j] == d[i][j]


def test_lattice_is_bipartite():
    for graph in graphs:
        print(graph)
        g = nx.Graph()
        for edge in graph.edges():
            g.add_edge(edge[0], edge[1])
        assert graph.is_bipartite() == nx.is_bipartite(g)


def test_lattice_is_connected():
    for graph in graphs:
        if graph.n_edges == 0:  # skip edgeless
            continue
        g = nx.Graph()
        for edge in graph.edges():
            g.add_edge(edge[0], edge[1])
        assert graph.is_connected() == nx.is_connected(g)


def test_adjacency_list():
    for graph in graphs:
        neigh = []
        g = nx.Graph()

        for i in range(graph.n_nodes):
            g.add_node(i)

        for edge in graph.edges():
            g.add_edge(edge[0], edge[1])
        for i in range(graph.n_nodes):
            neigh.append(set(g.neighbors(i)))
        dim = len(neigh)
        adl = graph.adjacency_list()

        for i in range(dim):
            assert set(adl[i]) in neigh


def test_grid_color_pbc():
    # compute length from iterator
    count = lambda it: sum(1 for _ in it)

    g = Grid([4, 4], pbc=True, color_edges=True)
    assert count(g.edges(filter_color=0)) == 16
    assert count(g.edges(filter_color=1)) == 16
    assert g.n_edges == 32

    g = Grid([4, 2], pbc=True, color_edges=True)
    assert count(g.edges(filter_color=0)) == 8
    assert count(g.edges(filter_color=1)) == 8

    g = Grid([4, 2], pbc=[True, False], color_edges=True)
    assert count(g.edges(filter_color=0)) == 8
    assert count(g.edges(filter_color=1)) == 4

    g = Grid([4, 2], pbc=False, color_edges=True)
    assert count(g.edges(filter_color=0)) == 6
    assert count(g.edges(filter_color=1)) == 4

    # with pytest.raises(ValueError, match="Directions with length <= 2 cannot have PBC"):
    #    g = Grid([2, 4], pbc=[True, True])

    g1 = Grid([7, 5], pbc=False)
    g2 = Grid([7, 5], pbc=[False, False])
    assert sorted(g1.edges()) == sorted(g2.edges())

    g1 = Grid([7, 5], pbc=True)
    g2 = Grid([7, 5], pbc=[True, True])
    assert sorted(g1.edges()) == sorted(g2.edges())


@pytest.mark.parametrize("graph", graphs)
def test_automorphisms(graph):
    if not graph.is_connected():
        return

    def is_automorphism(f, graph):
        edges = graph.edges()
        for v, w in edges:
            fv, fw = f[v], f[w]
            if (fv, fw) not in edges and (fw, fv) not in edges:
                print(f"E  ({(v, w)} -> {(fv, fw)})")
                return False
        return True

    autom = np.asarray(graph.automorphisms())
    for f in autom:
        assert is_automorphism(f, graph)


def _check_symmgroup(autom, symmgroup):
    """Asserts that symmgroup consists of automorphisms listed in autom and has no duplicate elements."""

    for el in symmgroup.to_array():
        assert group.Permutation(el) in autom.elems

    assert symmgroup == symmgroup.remove_duplicates()
    assert isinstance(symmgroup[0], group.Identity)


def _check_symmgroups(graph):
    autom = graph.automorphisms()
    _check_symmgroup(autom, graph.rotation_group())
    _check_symmgroup(autom, graph.point_group())
    _check_symmgroup(autom, graph.translation_group())
    _check_symmgroup(autom, graph.space_group())


def test_grid_translations():

    for ndim in 1, 2:
        g = Grid([4] * ndim, pbc=True)
        translations = g.translation_group()

        assert len(translations) == g.n_nodes

        _check_symmgroup(g.automorphisms(), translations)

        g = Grid([4] * ndim, pbc=False)
        translations = g.translation_group()
        assert translations.elems == [group.Identity()]  # only identity

    g = Grid([8, 4, 3], pbc=[True, False, False])
    assert len(g.translation_group()) == 8

    g = Grid([8, 4, 3], pbc=[True, True, False])
    assert len(g.translation_group()) == 8 * 4
    assert g.translation_group(2).elems == [group.Identity()]  # only identity

    g = Grid([8, 4, 3], pbc=[True, True, True])
    assert len(g.translation_group()) == 8 * 4 * 3
    assert len(g.translation_group(dim=0)) == 8
    assert len(g.translation_group(dim=1)) == 4
    assert len(g.translation_group(dim=2)) == 3

    t1 = g.translation_group()
    t2 = (
        g.translation_group(dim=0)
        @ g.translation_group(dim=1)
        @ g.translation_group(dim=2)
    )
    assert t1 == t2
    t2 = (
        g.translation_group(dim=2)
        @ g.translation_group(dim=1)
        @ g.translation_group(dim=0)
    )
    assert t1 != t2

    assert g.translation_group(dim=(0, 1)) == g.translation_group(
        0
    ) @ g.translation_group(1)


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
def test_grid_point_group_dim(n_dim):
    # point group of n-dimensional Hypercube should be the
    # hyperoctaherdal group of order 2^n n!, see
    # https://en.wikipedia.org/wiki/Hyperoctahedral_group
    point_group = Hypercube(length=3, n_dim=n_dim).point_group()
    order = 2**n_dim * math.factorial(n_dim)
    assert len(point_group) == order


def test_grid_space_group():

    g = nk.graph.Chain(8)
    _check_symmgroups(g)
    assert g.rotation_group().elems == [group.Identity()]
    assert len(g.point_group()) == 2  # one reflection
    assert len(g.space_group()) == 8 * 2  # translations * reflection

    g = nk.graph.Grid([8, 2], pbc=False)
    _check_symmgroups(g)
    assert len(g.rotation_group()) == 2
    assert len(g.point_group()) == 4
    assert g.point_group() == g.space_group()  # no PBC, no translations

    g = nk.graph.Grid([3, 3, 3], pbc=[True, True, False])
    _check_symmgroups(g)
    assert len(g.rotation_group()) == 8
    assert len(g.point_group()) == 16  # D_4 × Z_2
    assert len(g.space_group()) == 3 * 3 * 16

    g = nk.graph.Grid([3, 3, 3, 3], pbc=[True, True, False, False])
    _check_symmgroups(g)
    assert len(g.rotation_group()) == 32
    assert len(g.point_group()) == 64  # D_4 × D_4
    assert len(g.space_group()) == 3 * 3 * 64

    g = nk.graph.Hypercube(3, 2)
    _check_symmgroups(g)
    assert len(g.space_group()) == len(g.automorphisms())

    g = nk.graph.Hypercube(4, 2)
    _check_symmgroups(g)
    # 4x4 square has even higher symmetry
    assert len(g.space_group()) < len(g.automorphisms())


@pytest.mark.parametrize("lattice", [Triangular, Honeycomb, Kagome])
def test_triangular_space_group(lattice):
    g = lattice([3, 3])
    _check_symmgroups(g)
    assert len(g.rotation_group()) == 6
    assert len(g.point_group()) == 12
    assert len(g.space_group()) == 3 * 3 * 12

    g = lattice([3, 3], pbc=False)
    with pytest.raises(RuntimeError):
        _ = g.rotation_group()
    with pytest.raises(RuntimeError):
        _ = g.point_group()
    with pytest.raises(RuntimeError):
        _ = g.space_group()

    g = lattice([2, 4])
    with pytest.raises(RuntimeError):
        _ = g.rotation_group()
    with pytest.raises(RuntimeError):
        _ = g.point_group()
    with pytest.raises(RuntimeError):
        _ = g.space_group()
    # 2x4 unit cells of the triangle lattice make a rectangular grid
    assert len(g.point_group(group.planar.rectangle())) == 4


def test_kitaev_space_group():
    lattice = KitaevHoneycomb

    g = lattice([3, 3])
    _check_symmgroups(g)
    assert len(g.rotation_group()) == 2
    assert len(g.point_group()) == 2
    assert len(g.space_group()) == 3 * 3 * 2

    g = lattice([3, 3], pbc=False)
    with pytest.raises(RuntimeError):
        _ = g.rotation_group()
    with pytest.raises(RuntimeError):
        _ = g.point_group()
    with pytest.raises(RuntimeError):
        _ = g.space_group()

    g = lattice([2, 4])
    assert len(g.rotation_group()) == 2
    assert len(g.point_group()) == 2
    assert len(g.space_group()) == 2 * 4 * 2
    # 2x4 unit cells of the triangle lattice make a rectangular grid
    assert len(g.point_group(group.planar.rectangle())) == 4


def test_symmgroup():
    def assert_eq_hash(a, b):
        assert hash(a) == hash(b)
        assert a == b

    assert_eq_hash(group.Identity(), group.Identity())

    tr = Grid([8, 4, 3]).translation_group
    assert_eq_hash(tr(), tr(0) @ tr(1) @ tr(2))

    assert_eq_hash(tr().remove_duplicates(), tr())

    assert tr() @ tr() != tr()
    assert_eq_hash((tr() @ tr()).remove_duplicates(), tr())


def test_duplicate_atoms():
    lattice = Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[False, False],
        site_offsets=[[0, 0], [0, 0]],
    )
    np.testing.assert_almost_equal(lattice.site_offsets, np.array([[0, 0]]))


def test_edge_color_accessor():
    edges = [(0, 1, 0), (0, 3, 1), (1, 2, 1), (2, 3, 0)]
    g = Graph(edges=edges)

    assert edges == sorted(g.edges(return_color=True))

    g = Hypercube(4, 1)

    assert [(i, j, 0) for (i, j, _) in edges] == sorted(g.edges(return_color=True))


def test_union():
    graph1 = graphs[0]

    for graph in graphs:
        ug = nk.graph.disjoint_union(graph, graph1)

        assert ug.n_nodes == graph1.n_nodes + graph.n_nodes
        assert ug.n_edges == graph1.n_edges + graph.n_edges


def test_graph_conversions():
    igraph = ig.Graph.Star(6)
    g = Graph.from_igraph(igraph)
    assert g.n_nodes == igraph.vcount()
    assert g.edges() == igraph.get_edgelist()
    assert all(c == 0 for c in g.edge_colors)

    nxg = nx.star_graph(5)
    g = Graph.from_networkx(nxg)
    assert g.n_nodes == igraph.vcount()
    assert g.edges() == igraph.get_edgelist()
    assert all(c == 0 for c in g.edge_colors)

    igraph = ig.Graph()
    igraph.add_vertices(3)
    igraph.add_edges(
        [(0, 1), (1, 2)],
        attributes={
            "color": ["red", "blue"],
        },
    )
    with pytest.raises(ValueError, match="not all colors are integers"):
        _ = Graph.from_igraph(igraph)

    igraph = ig.Graph()
    igraph.add_vertices(3)
    igraph.add_edges(
        [(0, 1), (1, 2)],
        attributes={
            "color": [0, 1],
        },
    )
    g = Graph.from_igraph(igraph)
    assert g.edges(filter_color=0) == [(0, 1)]
    assert g.edges(filter_color=1) == [(1, 2)]
    assert g.edges(filter_color=0, return_color=True) == [(0, 1, 0)]
    assert g.edges(filter_color=1, return_color=True) == [(1, 2, 1)]

    with pytest.warns(FutureWarning):
        assert g.edges(color=0) == g.edges(filter_color=0)
    with pytest.warns(FutureWarning):
        assert g.edges(color=True) == g.edges(return_color=True)


def test_edge_colors():
    for g in graphs:
        assert all(isinstance(c, int) for c in g.edge_colors)


def test_lattice_k_neighbors():
    l0 = nk.graph.Chain(8, max_neighbor_order=1)
    l1 = nk.graph.Chain(8, max_neighbor_order=2)
    l2 = nk.graph.Chain(8, max_neighbor_order=3)

    colors = set(c for *_, c in l1.edges(return_color=True))
    assert colors == {0, 1}
    colors = set(c for *_, c in l2.edges(return_color=True))
    assert colors == {0, 1, 2}

    assert set(l0.edges()) == set(l1.edges(filter_color=0))
    assert set(l0.edges()) == set(l2.edges(filter_color=0))
    assert set(l1.edges(filter_color=1)) == set(l2.edges(filter_color=1))

    assert l0.n_edges < l1.n_edges < l2.n_edges

    with pytest.raises(RuntimeError, match="Lattice contains self-referential edge"):
        nk.graph.Chain(length=3, max_neighbor_order=3)

    for k in range(1, 11):
        assert nk.graph.Chain(100, max_neighbor_order=k).n_edges == 100 * k

    assert nk.graph.Square(10, pbc=True, max_neighbor_order=2).n_edges == 400

    g = nk.graph.Square(10, pbc=True, max_neighbor_order=2)
    assert len(g.edges(filter_color=0)) == 200
    assert len(g.edges(filter_color=1)) == 200
