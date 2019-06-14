# CustomGraph
A custom graph, specified by a list of edges and optionally colors.

## Class Constructor
Constructs a new graph given a list of edges.

|  Argument   |       Type       |                                                                                                                                                                                                                                     Description                                                                                                                                                                                                                                     |
|-------------|------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|edges        |iterable          |If `edges` has elements of type `Tuple[int, int]` it is treated as a list of edges. Then each element `(i, j)` means a connection between sites `i` and `j`. It is assumed that `0 <= i <= j`. Also, `edges` should contain no duplicates. If `edges` has elements of type `Tuple[int, int, int]` each element `(i, j, c)` represents an edge between sites `i` and `j` colored into `c`. It is again assumed that `0 <= i <= j` and that there are no duplicate elements in `edges`.|
|automorphisms|List[List[int]]=[]|The automorphisms of the graph, i.e. a List[List[int]] where the inner List[int] is a unique permutation of the graph sites.                                                                                                                                                                                                                                                                                                                                                         |

### Examples
A 10-site one-dimensional lattice with periodic boundary conditions can be
constructed specifying the edges as follows:

```python
>>> import netket
>>> g=netket.graph.CustomGraph([[i, (i + 1) % 10] for i in range(10)])
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
