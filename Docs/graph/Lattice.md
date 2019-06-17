# Lattice
A generic lattice built translating a unit cell and adding edges between nearest neighbours sites.
 The unit cell is defined by the ``basis_vectors`` and it can contain an arbitrary number of atoms.
 Each atom is located at an arbitrary position and is labelled by an integer number,
 meant to distinguish between the different atoms within the unit cell.
 Periodic boundary conditions can also be imposed along the desired directions.
 There are three different ways to refer to the lattice sites. A site can be labelled
 by a simple integer number (the site index), by its coordinates (actual position in space),
 or by a set of integers (the site vector), which indicates how many
 translations of each basis vectors have been performed while building the
 graph. The i-th component refers to translations along the i-th ``basis_vector`` direction.

## Class Constructor
Constructs a new ``Lattice`` given its side length and the features of the unit cell.

|  Argument   |        Type        |                                                                    Description                                                                    |
|-------------|--------------------|---------------------------------------------------------------------------------------------------------------------------------------------------|
|basis_vectors|List[List[float]]   |The basis vectors of the unit cell.                                                                                                                |
|extent       |List[int]           |The number of copies of the unit cell.                                                                                                             |
|pbc          |List[bool]=[]       |If ``True`` then the constructed lattice will have periodic boundary conditions, otherwise open boundary conditions are imposed (default=``True``).|
|atoms_coord  |List[List[float]]=[]|The coordinates of different atoms in the unit cell (default=one atom at the origin).                                                              |

### Examples
Constructs a rectangular 3X4 lattice with periodic boundary conditions.

```python
>>> import netket
>>> g=netket.graph.Lattice(basis_vectors=[[1,0],[0,1]],extent=[3,4])
>>> print(g.n_sites)
12

```



## Class Methods 
### atom_label
Member function returning the atom label indicating which of the unit cell atoms is located at a given a site index.

|Argument|Type|  Description  |
|--------|----|---------------|
|site    |int |The site index.|

### site_to_coord
Member function returning the coordinates of a given site index.

|Argument|Type|  Description  |
|--------|----|---------------|
|site    |int |The site index.|

### site_to_vector
Member function returning the site vector corresponding to a given site index.

|Argument|Type|  Description  |
|--------|----|---------------|
|site    |int |The site index.|

### Examples
Constructs a square 2X2 lattice without periodic boundary conditions and prints the site vectors corresponding to given site indices.

```python
 >>> import netket
 >>> g=netket.graph.Lattice(basis_vectors=[[1.,0.],[0.,1.]], extent=[2,2], pbc=[0,0])
 >>> print(list(map(int,g.site_to_vector(0))))
 [0, 0]
 >>> print(list(map(int,g.site_to_vector(1))))
 [0, 1]
 >>> print(list(map(int,g.site_to_vector(2))))
 [1, 0]
 >>> print(list(map(int,g.site_to_vector(3))))
 [1, 1]

  ```

### vector_to_coord
Member function returning the coordinates of a given atom characterized by a
given site vector.

| Argument  |  Type   |            Description             |
|-----------|---------|------------------------------------|
|site_vector|List[int]|The site vector.                    |
|atom_label |int      |Which of the atoms in the unit cell.|

### vector_to_site
Member function returning the site index corresponding to a given site vector.

| Argument  |  Type   |  Description   |
|-----------|---------|----------------|
|site_vector|List[int]|The site vector.|

## Properties

|   Property   |      Type       |                                                        Description                                                        |
|--------------|-----------------|---------------------------------------------------------------------------------------------------------------------------|
|adjacency_list|       list      | The adjacency list of the graph where each node is           represented by an integer in `[0, n_sites)`.                 |
|automorphisms |       list[list]| The automorphisms of the graph,           including translation symmetries only.                                          |
|basis_vectors |       list[list]| The basis vectors of the lattice.                                                                                         |
|coordinates   |       list[list]| The coordinates of the atoms in the lattice.                                                                              |
|distances     |       list[list]| The distances between the nodes. The fact that some node           may not be reachable from another is represented by -1.|
|edges         |       list      | The graph edges.                                                                                                          |
|is_bipartite  |       bool      | Whether the graph is bipartite.                                                                                           |
|is_connected  |       bool      | Whether the graph is connected.                                                                                           |
|n_dim         |       int       | The dimension of the lattice.                                                                                             |
|n_sites       |       int       | The number of vertices in the graph.                                                                                      |
