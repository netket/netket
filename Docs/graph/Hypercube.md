# Hypercube
A hypercube lattice of side L in d dimensions.
 Periodic boundary conditions can also be imposed.

## Class Constructor [1]
Constructs a new ``Hypercube`` given its side length and dimension.

|Argument|  Type   |                                                           Description                                                            |
|--------|---------|----------------------------------------------------------------------------------------------------------------------------------|
|length  |int      |Side length of the hypercube. It must always be >=1, but if ``pbc==True`` then the minimal valid length is 3.                     |
|n_dim   |int=1    |Dimension of the hypercube. It must be at least 1.                                                                                |
|pbc     |bool=True|If ``True`` then the constructed hypercube will have periodic boundary conditions, otherwise open boundary conditions are imposed.|

### Examples
A 10x10 square lattice with periodic boundary conditions can be
constructed as follows:

```python
>>> import netket
>>> g=netket.graph.Hypercube(length=10,n_dim=2,pbc=True)
>>> print(g.n_sites)
100

```


## Class Constructor [2]
Constructs a new `Hypercube` given its side length and edge coloring.

|Argument|  Type  |                                                                                 Description                                                                                 |
|--------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|length  |int     |Side length of the hypercube. It must always be >=3 if the hypercube has periodic boundary conditions and >=1 otherwise.                                                     |
|colors  |iterable|Edge colors, must be an iterable of `Tuple[int, int, int]` where each element `(i, j, c) represents an edge `i <-> j` of color `c`. Colors must be assigned to **all** edges.|

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
