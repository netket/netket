import pytest

import netket as nk
import networkx as nx
import math
import numpy as np
import igraph as ig

from netket.graph import *

from .. import common

pytestmark = common.skipif_mpi

nxg = nx.star_graph(10)
graphs = [
    # star graph
    Graph(edges=list(nxg.edges())),
    # Grid graphs
    Hypercube(length=10, n_dim=1, pbc=True),
    Hypercube(length=4, n_dim=2, pbc=True),
    Hypercube(length=5, n_dim=1, pbc=False),
    Grid(length=[2, 2], pbc=False),
    Grid(length=[4, 2], pbc=[True, False]),
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

symmetric_graphs = [
    # Square
    nk.graph.Lattice(basis_vectors=[[0, 1], [1, 0]], extent=[3, 3]),
    # Triangular
    nk.graph.Lattice(basis_vectors=[[0, 1], [np.sqrt(3) / 2, 1 / 2]], extent=[3, 3]),
    # Honeycomb
    nk.graph.Lattice(
        basis_vectors=[[0, 1], [np.sqrt(3) / 2, 1 / 2]],
        site_offsets=[[1 / (2 * np.sqrt(3)), 1 / 2], [1 / np.sqrt(3), 1]],
        extent=[3, 3],
    ),
    # Kagome
    nk.graph.Lattice(
        basis_vectors=[[0, 1], [np.sqrt(3) / 2, 1 / 2]],
        site_offsets=[[0, 1 / 2], [np.sqrt(3) / 4, 1 / 4], [np.sqrt(3) / 4, 3 / 4]],
        extent=[3, 3],
    ),
    # Cube
    nk.graph.Lattice(basis_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]], extent=[2, 2, 2]),
    # Body Centered Cubic
    nk.graph.Lattice(
        basis_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        site_offsets=[[0, 0, 0], [1 / 2, 1 / 2, 1 / 2]],
        extent=[2, 2, 2],
    ),
]

unit_cells = [9, 9, 9, 9, 8, 8]

atoms_per_unit_cell = [1, 1, 2, 3, 1, 2]

coordination_number = [4, 6, 3, 4, 6, 8]

rot_symmetry = [4, 6, 6, 6, 4, 4]


def test_lattice_graphs():
    # Check to see if graphs have the correct number of nodes and edges
    for i, graph in enumerate(symmetric_graphs):
        assert graph.n_nodes == unit_cells[i] * atoms_per_unit_cell[i]
        assert graph.n_edges == graph.n_nodes * coordination_number[i] // 2


def test_lattice():
    for g in graphs + symmetric_graphs:
        if not isinstance(g, Lattice):
            continue

        # sites should be sorted in lexicographic order by basis coordinate
        sort = np.lexsort(g.basis_coords.T[::-1])
        print(g.basis_coords[sort])
        assert np.all(sort == np.arange(g.n_nodes))

        # check lookup with id
        for i, site_id in enumerate(g.nodes()):
            assert i == site_id
            cc = g.basis_coords[i]
            pos = g.positions[i]
            manual_pos = g.basis_vectors.T @ cc[:-1] + g.site_offsets[cc[-1]]
            assert np.allclose(manual_pos, pos)

            assert g.id_from_position(pos) == i
            assert g.id_from_basis_coords(cc) == i

        # check lookup with arrays
        if g.n_nodes > 1:
            pos = g.positions[[0, 1]]
            ids = g.id_from_position(pos)
            # assert ids.ndim == 1 and ids.size == 2
            assert np.all(ids == [0, 1])

            ccs = g.basis_coords[[0, 1]]
            ids = g.id_from_basis_coords(ccs)
            # assert ids.ndim == 1 and ids.size == 2
            assert np.all(ids == [0, 1])

            pos2 = g.position_from_basis_coords(ccs)
            assert np.all(pos2 == pos)

    g = Lattice([[1]], [2])
    pos = [[0.0], [1.0], [0.5]]
    ids = g.id_from_position(pos)
    assert np.all(ids[:2] == [0, 1])
    assert ids[2] is None

    with pytest.raises(KeyError):
        pos = g.position_from_basis_coords([2])
    with pytest.raises(KeyError):
        pos = g.position_from_basis_coords([[2]])


def test_lattice_old_interface():
    with pytest.warns(FutureWarning):
        _ = Lattice(basis_vectors=[[1.0]], atoms_coord=[[0.0], [0.5]], extent=[4])

    def check_alternative(method, alternative):
        with pytest.warns(FutureWarning):
            result = method()
        assert np.all(alternative() == result)

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


def test_lattice_symmetry():
    # Check if the symmetry groups make sense
    for i, graph in enumerate(symmetric_graphs):
        rots = np.asarray(graph.planar_rotations(rot_symmetry[i]))
        # R(2pi/period)*R(2pi(1-1/period) = I)
        assert np.all(rots[-1][rots[1]] == np.arange(graph.n_nodes))
        # Try a bad rotation and fail
        with pytest.raises(ValueError):
            graph.planar_rotations(5)

        # Two reflections returns graph to itself
        ref = np.asarray(graph.reflections())
        assert np.all(ref[1][ref[1]] == np.arange(graph.n_nodes))

        # T(0,1)T(0,-1) = I
        translations = np.asarray(graph.basis_translations())
        if unit_cells[i] == 8:
            # [1,0,0] maps to itself
            assert np.all(translations[1][translations[1]] == np.arange(graph.n_nodes))
            # [1,1,1] maps to itself
            assert np.all(
                translations[-1][translations[-1]] == np.arange(graph.n_nodes)
            )
        if unit_cells[i] == 9:
            # T(2,0)T(1,0) maps to itself
            assert np.all(translations[2][translations[1]] == np.arange(graph.n_nodes))
            # T(2,1)T(1,2) maps to itself
            assert np.all(translations[7][translations[5]] == np.arange(graph.n_nodes))


def coord2index(xs, length):
    if isinstance(xs, int):
        return xs
    i = 0
    scale = 1
    for x in xs:
        i += scale * x
        scale *= length
    return i


def check_edges(length, n_dim, pbc):
    x = nx.grid_graph(dim=[length] * n_dim, periodic=pbc)
    x_edges = [[coord2index(i, length) for i in edge] for edge in x.edges()]
    x_edges = sorted([sorted(ed) for ed in x_edges])
    y = nk.graph.Hypercube(length=length, n_dim=n_dim, pbc=pbc)
    y_edges = sorted([sorted(ed) for ed in y.edges()])
    assert x_edges == y_edges


def test_graph_wrong():
    with pytest.raises(TypeError):
        nk.graph.Graph(5)

    with pytest.raises(TypeError):
        nk.graph.Graph([1, 2, 3], True)

    with pytest.raises(ValueError):
        nk.graph.Graph([1, 2, 3], [1, 2, 3])


def test_edges_are_correct():
    check_edges(1, 1, False)
    check_edges(1, 2, False)
    for length in [3, 4, 5]:
        for dim in [1, 2, 3]:
            for pbc in [True, False]:
                check_edges(length, dim, pbc)
    for pbc in [True, False]:
        check_edges(3, 7, pbc)


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
            ax = lattice.draw(
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
            y = nk.graph.Graph(nodes=list(x.nodes()), edges=list(x.edges()))

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
            y = nk.graph.Graph(nodes=list(x.nodes()), edges=list(x.edges()))
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
    assert count(g.edges(color=0)) == 16
    assert count(g.edges(color=1)) == 16
    assert g.n_edges == 32

    g = Grid([4, 2], pbc=True, color_edges=True)
    assert count(g.edges(color=0)) == 8
    assert count(g.edges(color=1)) == 4

    g = Grid([4, 2], pbc=False, color_edges=True)
    assert count(g.edges(color=0)) == 6
    assert count(g.edges(color=1)) == 4

    with pytest.raises(ValueError, match="Directions with length <= 2 cannot have PBC"):
        g = Grid([2, 4], pbc=[True, True])

    g1 = Grid([7, 5], pbc=False)
    g2 = Grid([7, 5], pbc=[False, False])
    assert sorted(g1.edges()) == sorted(g2.edges())

    g1 = Grid([7, 5], pbc=True)
    g2 = Grid([7, 5], pbc=[True, True])
    assert sorted(g1.edges()) == sorted(g2.edges())


# skip star graph because it has 10! (≈ 3 million) isomorphisms
@pytest.mark.parametrize("graph", graphs[1:])
def test_automorphisms(graph):
    if not graph.is_connected():
        return

    g = ig.Graph(edges=graph.edges())
    autom = g.get_isomorphisms_vf2()
    autom_g = graph.automorphisms()
    dim = len(autom_g)
    for i in range(dim):
        assert np.asarray(autom_g[i]).tolist() in autom


def _check_symmgroup(graph, symmgroup):
    """Asserts that symmgroup consists of automorphisms and has no duplicate elements."""
    from netket.utils.semigroup import Permutation

    autom = graph.automorphisms()
    for el in symmgroup.to_array():
        assert Permutation(el) in autom.elems

    assert symmgroup == symmgroup.remove_duplicates()


def test_grid_translations():
    from netket.utils.semigroup import Identity
    from netket.graph.grid import Translation

    for ndim in 1, 2:
        g = Grid([4] * ndim, pbc=True)
        translations = g.translations()

        assert len(translations) == g.n_nodes

        _check_symmgroup(g, translations)

        g = Grid([4] * ndim, pbc=False)
        translations = g.translations()
        assert translations.elems == [Identity()]  # only identity

    g = Grid([8, 4, 3], pbc=[True, False, False])
    assert len(g.translations()) == 8

    g = Grid([8, 4, 3], pbc=[True, True, False])
    assert len(g.translations()) == 8 * 4
    with pytest.raises(ValueError):
        g.translations(dim=2)  # no translation symmetry along non-periodic dim

    g = Grid([8, 4, 3], pbc=[True, True, True])
    assert len(g.translations()) == 8 * 4 * 3
    assert len(g.translations(dim=0)) == 8
    assert len(g.translations(dim=1)) == 4
    assert len(g.translations(dim=2)) == 3
    assert len(g.translations(dim=0, step=2)) == 4
    assert len(g.translations(dim=0, step=4) @ g.translations(dim=2)) == 6

    t1 = g.translations()
    t2 = g.translations(dim=0) @ g.translations(dim=1) @ g.translations(dim=2)
    assert t1 == t2
    t2 = g.translations(dim=2) @ g.translations(dim=1) @ g.translations(dim=0)
    assert t1 != t2

    assert g.translations(dim=(0, 1)) == g.translations(0) @ g.translations(1)

    assert Translation((1,), (2,)) @ Translation((1,), (2,)) == Translation((2,), (2,))

    with pytest.raises(ValueError, match="Incompatible translations"):
        Translation((1,), (2,)) @ Translation((1,), (8,))


@pytest.mark.parametrize("n_dim", [1, 2, 3, 4])
def test_grid_space_group_dim(n_dim):
    # space group of n-dimensional Hypercube should be the
    # hyperoctaherdal group of order 2^n n!, see
    # https://en.wikipedia.org/wiki/Hyperoctahedral_group
    space_group = Hypercube(length=3, n_dim=n_dim).space_group()
    order = 2 ** n_dim * math.factorial(n_dim)
    assert len(space_group) == order


def test_grid_space_group():
    def _check_symmgroups(g):
        _check_symmgroup(g, g.rotations())
        _check_symmgroup(g, g.space_group())
        _check_symmgroup(g, g.lattice_group())

    from netket.utils.semigroup import Identity

    g = nk.graph.Chain(8)
    _check_symmgroups(g)
    assert g.rotations().elems == [Identity()]
    assert len(g.space_group()) == 2  # one reflection
    assert g.space_group() == g.axis_reflection() == g.axis_reflection(0)
    with pytest.raises(ValueError):  # invalid axis
        g.axis_reflection(1)
    assert len(g.lattice_group()) == 8 * 2  # translations * reflection

    g = nk.graph.Grid([8, 2], pbc=False)
    _check_symmgroups(g)
    assert len(g.rotations()) == 2  # one 180 deg rotation
    assert len(g.space_group()) == 4
    assert g.lattice_group() == g.space_group()  # no PBC, no translations

    g = nk.graph.Grid([5, 4, 3], pbc=[True, False, False])
    _check_symmgroups(g)
    rot1 = g.rotations(remove_duplicates=False)
    rot2 = g.rotations(remove_duplicates=True)
    assert len(rot1) > len(rot2)
    rot3, inverse = rot1.remove_duplicates(return_inverse=True)
    assert rot2 == rot3
    assert np.all(rot3.to_array()[inverse] == rot1.to_array())

    g = nk.graph.Hypercube(3, 2)
    _check_symmgroups(g)
    assert len(g.lattice_group()) == len(g.automorphisms())

    g = nk.graph.Hypercube(4, 2)
    _check_symmgroups(g)
    # 4x4 cube has even higher symmetry
    assert len(g.lattice_group()) < len(g.automorphisms())


def test_symmgroup():
    from netket.utils.semigroup import Identity

    def assert_eq_hash(a, b):
        assert hash(a) == hash(b)
        assert a == b

    assert_eq_hash(Identity(), Identity())

    tr = Grid([8, 4, 3]).translations
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
    assert np.all(lattice.site_offsets == np.array([[0, 0]]))


# def test_edge_color_accessor():
#     edges = [(0, 1, 0), (1, 2, 1), (2, 3, 0), (0, 3, 1)]
#     g = Graph(edges)
#
#     assert edges == sorted(g.edges(color=True))
#
#     g = Hypercube(4, 1)
#
#     assert [(i, j, 0) for (i, j, _) in edges] == sorted(g.edge_colors)


def test_union():
    graph1 = graphs[0]

    for graph in graphs:
        ug = nk.graph.disjoint_union(graph, graph1)

        assert ug.n_nodes == graph1.n_nodes + graph.n_nodes
        assert ug.n_edges == graph1.n_edges + graph.n_edges
