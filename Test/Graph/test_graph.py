import netket as nk
import networkx as nx
import math
import numpy as np
import igraph as ig

from netket.graph import *

import pytest

nxg = nx.star_graph(10)
graphs = [
    Hypercube(length=10, n_dim=1, pbc=True),
    Hypercube(length=4, n_dim=2, pbc=True),
    Hypercube(length=5, n_dim=1, pbc=False),
    Grid(length=[2, 2], pbc=False),
    Grid(length=[4, 2], pbc=[True, False]),
    Graph(edges=list(nxg.edges())),
    Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[False, False],
        atoms_coord=[[0, 0]],
    ),
    Lattice(
        basis_vectors=[[1.5, math.sqrt(3) / 2.0], [0, math.sqrt(3)]],
        extent=[3, 5],
        atoms_coord=[[0, 0], [1, 1]],
    ),
    Lattice(
        basis_vectors=[[2.0, 0.0], [1.0, math.sqrt(3)]],
        extent=[4, 4],
        atoms_coord=[[0, 0], [1.0 / 2.0, math.sqrt(3) / 2.0], [1.0, 0.0]],
    ),
    Lattice(
        basis_vectors=[
            [1.0, 0.0, 0.0],
            [1.0 / 2.0, math.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        extent=[6, 7, 4],
        atoms_coord=[[0, 0, 0]],
    ),
    Edgeless(10),
]
lattices = [
    Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[False, False],
        atoms_coord=[[0, 0]],
    ),
    Lattice(
        basis_vectors=[[1.5, math.sqrt(3) / 2.0], [0, math.sqrt(3)]],
        extent=[3, 5],
        atoms_coord=[[0, 0], [1, 1]],
    ),
    Lattice(
        basis_vectors=[[2.0, 0.0], [1.0, math.sqrt(3)]],
        extent=[4, 4],
        atoms_coord=[[0, 0], [1.0 / 2.0, math.sqrt(3) / 2.0], [1.0, 0.0]],
    ),
    Lattice(
        basis_vectors=[
            [1.0, 0.0, 0.0],
            [1.0 / 2.0, math.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        extent=[6, 7, 4],
        atoms_coord=[[0, 0, 0]],
    ),
]


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
    for graph in lattices:
        g = nx.Graph()
        for edge in graph.edges():
            g.add_edge(edge[0], edge[1])
        assert graph.is_bipartite() == nx.is_bipartite(g)


def test_lattice_is_connected():
    for graph in lattices:
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


def test_automorphisms():
    for graph in lattices:
        if graph.is_connected():  # to avoid troubles with ig automorphisms
            g = ig.Graph(edges=graph.edges())
            autom = g.get_isomorphisms_vf2()
            autom_g = graph.automorphisms()
            dim = len(autom_g)
            for i in range(dim):
                assert autom_g[i] in autom


def test_grid_translations():
    from netket.utils.semigroup import Identity

    for ndim in 1, 2:
        g = Grid([4] * ndim, pbc=True)
        translations = g.translations()

        assert len(translations) == g.n_nodes

        autom = g.automorphisms()
        for t in translations.indices().tolist():
            assert t in autom

        g = Grid([4] * ndim, pbc=False)
        translations = g.translations()
        assert translations.elems == [Identity()]  # only identity

    g = Grid([8, 4, 3], pbc=[True, False, False])
    assert len(g.translations()) == 8

    g = Grid([8, 4, 3], pbc=[True, True, False])
    assert len(g.translations()) == 8 * 4

    g = Grid([8, 4, 3], pbc=[True, True, True])
    assert len(g.translations()) == 8 * 4 * 3
    assert len(g.translations(dim=0)) == 8
    assert len(g.translations(dim=1)) == 4
    assert len(g.translations(dim=2)) == 3
    assert len(g.translations(dim=0, period=2)) == 4

    t1 = g.translations()
    t2 = g.translations(dim=0) @ g.translations(dim=1) @ g.translations(dim=2)
    assert np.all(t1.indices() == t2.indices())


def test_duplicate_atoms():
    lattice = Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[False, False],
        atoms_coord=[[0, 0], [0, 0]],
    )
    assert np.all(lattice.atoms_coord == np.array([[0, 0]]))


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
    graph1 = lattices[0]

    for graph in lattices:
        ug = nk.graph.disjoint_union(graph, graph1)

        assert ug.n_nodes == graph1.n_nodes + graph.n_nodes
        assert ug.n_edges == graph1.n_edges + graph.n_edges
