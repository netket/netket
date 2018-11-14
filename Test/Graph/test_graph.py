import netket as nk
import networkx as nx
import numpy as np
from mpi4py import MPI
nxg = nx.star_graph(10)
graphs = [
    nk.graph.Hypercube(L=10, ndim=1, pbc=True),
    nk.graph.Hypercube(L=4, ndim=2, pbc=True),
    nk.graph.CustomGraph(nxg.edges())
]


def tonx(graph):
    adl = graph.AdjacencyList()
    i = 0
    edges = []
    for els in adl:
        for el in els:
            edges.append([i, el])
        i += 1
    if edges:
        return nx.from_edgelist(edges)

    gx = nx.Graph()
    for i in range(graph.Nsites()):
        gx.add_node(i)
    return gx


def test_size_is_positive():
    for graph in graphs:
        assert graph.Nsites() > 0


def test_is_connected():
    for graph in graphs:
        assert graph.IsConnected() == nx.is_connected(tonx(graph))


def test_computes_distances():
    for graph in graphs:
        if (graph.IsConnected()):
            nxg = tonx(graph)
            d = graph.Distances(0)
            d1 = nx.shortest_path_length(nxg, source=0)
            for dist, dist1 in zip(d, d1):
                assert dist == dist1

            d = graph.AllDistances()
            d1 = dict(nx.shortest_path_length(nxg))
            for i in range(graph.Nsites()):
                for j in range(graph.Nsites()):
                    assert d1[i][j] == d[i][j]
