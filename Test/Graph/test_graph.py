import netket as nk
import networkx as nx
from mpi4py import MPI

nxg = nx.star_graph(10)
graphs = [
    nk.graph.Hypercube(length=10, n_dim=1, pbc=True),
    nk.graph.Hypercube(length=4, n_dim=2, pbc=True),
    nk.graph.Hypercube(length=5, n_dim=1, pbc=False),
    nk.graph.CustomGraph(nxg.edges())
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
    x = nx.grid_graph(dim=[length]*n_dim, periodic=pbc)
    x_edges = sorted([[coord2index(i, length) for i in edge] for edge in x.edges])
    y = nk.graph.Hypercube(length=length, n_dim=n_dim, pbc=pbc)
    y_edges = sorted(list(y.edges()))
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
    for i in range(graph.n_sites()):
        gx.add_node(i)
    return gx

def test_size_is_positive():
    for graph in graphs:
        assert graph.n_sites() > 0

def test_is_connected():
    for i in range(5, 10):
        for j in range(5, 30, 5):
            x = nx.dense_gnm_random_graph(i, j)
            y = nk.graph.CustomGraph(x.edges())
            assert y.is_connected == nx.is_connected(x)

def test_computes_distances():
    for graph in graphs:
        if (graph.is_connected):
            nxg = tonx(graph)
            d = graph.distances()
            d1 = dict(nx.shortest_path_length(nxg))
            for i in range(graph.n_sites()):
                for j in range(graph.n_sites()):
                    assert d1[i][j] == d[i][j]
