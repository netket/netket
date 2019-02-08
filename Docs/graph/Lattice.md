# Lattice
A generic lattice built translating a unit cell. The unit cell can contain an arbitrary number of atoms, located at arbitrary positions. Periodic boundary conditions can also be imposed along the desired directions.

## Class Constructor
Constructs a new ``Lattice`` given its side length and the features of the unit cell.

|  Argument   |        Type        |                                                                  Description                                                                  |
|-------------|--------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
|basis_vectors|List[List[float]]   |The basis vectors of the unit cell.                                                                                                            |
|extent       |List[int]           |The number of copies of the unit cell.                                                                                                         |
|pbc          |List[bool]=[]       |If ``True`` then the constructed lattice will have periodic boundary conditions, otherwise open boundary conditions are imposed (default=True).|
|atoms_coord  |List[List[float]]=[]|The coordinates of different atoms in the unit cell (default=one atom at the origin).                                                          |

### Examples
Constructs a rectangular 3X4 lattice with periodic boundaries

```python
>>> from netket.graph import Lattice
>>> g=Lattice(basis_vectors=[[1,0],[0,1]],extent=[3,4])
>>> print(g.n_sites)
12

```



## Class Methods 
## Properties

|   Property   |           Type           |                                                        Description                                                        |
|--------------|--------------------------|---------------------------------------------------------------------------------------------------------------------------|
|adjacency_list|       list               | The adjacency list of the graph where each node is           represented by an integer in `[0, n_sites)`                  |
|automorphisms |       list[list]         | The automorphisms of the graph,           including translation symmetries only.                                          |
|coordinates   |                list[list]| The coordinates of the atoms in the lattice.                                                                              |
|distances     |       list[list]         | The distances between the nodes. The fact that some node           may not be reachable from another is represented by -1.|
|edges         |       list               | The graph edges.                                                                                                          |
|is_bipartite  |       bool               | Whether the graph is bipartite.                                                                                           |
|is_connected  |       bool               | Whether the graph is connected.                                                                                           |
|n_dim         |                int       | The dimension of the lattice.                                                                                             |
|n_sites       |       int                | The number of vertices in the graph.                                                                                      |
