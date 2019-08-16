import netket as nk
import networkx as nx
import igraph as ig
import math

nxg = nx.star_graph(10)
graphs = [
    nk.graph.Hypercube(length=10, n_dim=1, pbc=True),
    nk.graph.Hypercube(length=4, n_dim=2, pbc=True),
    nk.graph.Hypercube(length=5, n_dim=1, pbc=False),
    nk.graph.CustomGraph(nxg.edges()),
    nk.graph.Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[0, 0],
        atoms_coord=[[0, 0]],
    ),
    nk.graph.Lattice(
        basis_vectors=[[1.5, math.sqrt(3) / 2.0], [0, math.sqrt(3)]],
        extent=[3, 5],
        atoms_coord=[[0, 0], [1, 0]],
    ),
    nk.graph.Lattice(
        basis_vectors=[[2.0, 0.0], [1.0, math.sqrt(3)]],
        extent=[4, 4],
        atoms_coord=[[0, 0], [1.0 / 2.0, math.sqrt(3) / 2.0], [1.0, 0.0]],
    ),
    nk.graph.Lattice(
        basis_vectors=[
            [1.0, 0.0, 0.0],
            [1.0 / 2.0, math.sqrt(3) / 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        extent=[6, 7, 4],
        atoms_coord=[[0, 0, 0]],
    ),
]
lattices = [
    nk.graph.Lattice(
        basis_vectors=[[1.0, 0.0], [1.0 / 2.0, math.sqrt(3) / 2.0]],
        extent=[10, 10],
        pbc=[0, 0],
        atoms_coord=[[0, 0]],
    ),
    nk.graph.Lattice(
        basis_vectors=[[1.5, math.sqrt(3) / 2.0], [0, math.sqrt(3)]],
        extent=[3, 5],
        atoms_coord=[[0, 0], [1, 0]],
    ),
    nk.graph.Lattice(
        basis_vectors=[[2.0, 0.0], [1.0, math.sqrt(3)]],
        extent=[4, 4],
        atoms_coord=[[0, 0], [1.0 / 2.0, math.sqrt(3) / 2.0], [1.0, 0.0]],
    ),
    nk.graph.Lattice(
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
    x_edges = [[coord2index(i, length) for i in edge] for edge in x.edges]
    x_edges = sorted([sorted(ed) for ed in x_edges])
    y = nk.graph.Hypercube(length=length, n_dim=n_dim, pbc=pbc)
    y_edges = sorted([sorted(ed) for ed in y.edges])
    assert x_edges == y_edges


def test_edges_are_correct():
    check_edges(1, 1, False)
    check_edges(1, 2, False)
    for length in [3, 4, 5]:
        for dim in [1, 2, 3]:
            for pbc in [True, False]:
                check_edges(length, dim, pbc)
    for pbc in [True, False]:
        check_edges(3, 7, pbc)


def tonx(graph):
    adl = graph.adjacency_list
    i = 0
    edges = []
    for els in adl:
        for el in els:
            edges.append([i, el])
        i += 1
    if edges:
        return nx.from_edgelist(edges)

    gx = nx.Graph()
    for i in range(graph.n_sites):
        gx.add_node(i)
    return gx


def test_size_is_positive():
    for graph in graphs:
        assert graph.n_sites > 0


def test_is_connected():
    for i in range(5, 10):
        for j in range(i + 1, i * i):
            x = nx.dense_gnm_random_graph(i, j)
            y = nk.graph.CustomGraph(x.edges)
            if len(x) == len(
                set((i for (i, j) in x.edges)) | set((j for (i, j) in x.edges))
            ):
                assert y.is_connected == nx.is_connected(x)
            else:
                assert not nx.is_connected(x)


def test_is_bipartite():
    for i in range(1, 10):
        for j in range(1, i * i):
            x = nx.dense_gnm_random_graph(i, j)
            y = nk.graph.CustomGraph(x.edges)
            # if len(x) == len(set((i for (i, j) in x.edges)) | set((j for (i, j) in x.edges))):
            assert y.is_bipartite == nx.is_bipartite(x)
            # else:
            # assert not nx.is_bipartite(x)


def test_computes_distances():
    for graph in graphs:
        if graph.is_connected:
            nxg = nx.from_edgelist(graph.edges)
            d = graph.distances
            d1 = dict(nx.shortest_path_length(nxg))
            for i in range(graph.n_sites):
                for j in range(graph.n_sites):
                    assert d1[i][j] == d[i][j]


def test_lattice_is_bipartite():
    for graph in lattices:
        g = nx.Graph()
        for edge in graph.edges:
            g.add_edge(edge[0], edge[1])
        assert graph.is_bipartite == nx.is_bipartite(g)


def test_lattice_is_connected():
    for graph in lattices:
        g = nx.Graph()
        for edge in graph.edges:
            g.add_edge(edge[0], edge[1])
        assert graph.is_connected == nx.is_connected(g)


def test_adjacency_list():
    for graph in graphs:
        neigh = []
        g = nx.Graph()
        for edge in graph.edges:
            g.add_edge(edge[0], edge[1])
        for i in range(graph.n_sites):
            neigh.append(set(g.neighbors(i)))
        dim = len(neigh)
        for i in range(dim):
            assert set(graph.adjacency_list[i]) in neigh


def test_automorphisms():
    for graph in lattices:
        if graph.is_connected:  # for not to have troubles with ig automorphisms
            g = ig.Graph(edges=graph.edges)
            autom = g.get_isomorphisms_vf2()
            dim = len(graph.automorphisms)
            for i in range(dim):
                assert graph.automorphisms[i] in autom
