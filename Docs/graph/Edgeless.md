# Edgeless
A set graph, i.e. a collection of unconnected vertices.

## Class Constructor
Constructs a new set of given number of vertices.

| Argument |Type|      Description      |
|----------|----|-----------------------|
|n_vertices|int |The number of vertices.|

### Examples
A 10-site set:

```python
>>> import netket
>>> g=netket.graph.Edgeless(10)
>>> print(g.n_sites)
10

```



## Class Methods 
## Properties

|   Property   |      Type       |                                                        Description                                                        |
|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
|adjacency_list|       list      | The adjacency list of the graph where each node is           represented by an integer in `[0, n_sites)`.                 |
|automorphisms |       list[list]| The automorphisms of the graph,           including translation symmetries only.                                          |
|distances     |       list[list]| The distances between the nodes. The fact that some node           may not be reachable from another is represented by -1.|
|edges         |       list      | The graph edges.                                                                                                          |
|is_bipartite  |       bool      | Whether the graph is bipartite.                                                                                           |
|is_connected  |       bool      | Whether the graph is connected.                                                                                           |
|n_sites       |       int       | The number of vertices in the graph.                                                                                      |
