# GraphOperator
A custom graph based operator.

## Class Constructor
Constructs a new ``GraphOperator`` given a hilbert space and either a
list of operators acting on sites or a list acting on the bonds.
Users can specify the color of the bond that an operator acts on, if
desired. If none are specified, the bond operators act on all edges.

|   Argument   |              Type              |                                                                               Description                                                                               |
|--------------|--------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|hilbert       |netket._C_netket.hilbert.Hilbert|Hilbert space the operator acts on.                                                                                                                                      |
|siteops       |List[List[List[complex]]]=[]    |A list of operators that act on the nodes of the graph. The default is an empty list. Note that if no siteops are specified, the user must give a list of bond operators.|
|bondops       |List[List[List[complex]]]=[]    |A list of operators that act on the edges of the graph. The default is an empty list. Note that if no bondops are specified, the user must give a list of site operators.|
|bondops_colors|List[int]=[]                    |A list of edge colors, specifying the color each bond operator acts on. The defualt is an empty list.                                                                    |

### Examples
Constructs a ``BosGraphOperator`` operator for a 2D system.

```python
>>> import netket as nk
>>> sigmax = [[0, 1], [1, 0]]
>>> mszsz = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
>>> edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8],
... [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15],
... [15, 16], [16, 17], [17, 18], [18, 19], [19, 0]]
>>> g = nk.graph.CustomGraph(edges=edges)
>>> hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], graph=g)
>>> op = nk.operator.GraphOperator(
... hi, siteops=[sigmax], bondops=[mszsz], bondops_colors=[0])
>>> print(op.hilbert.size)
20

```



## Class Methods 
### get_conn
Member function finding the connected elements of the Operator. Starting
from a given visible state v, it finds all other visible states v' such
that the matrix element O(v,v') is different from zero. In general there
will be several different connected visible units satisfying this
condition, and they are denoted here v'(k), for k=0,1...N_connected.

|Argument|            Type            |                   Description                    |
|--------|----------------------------|--------------------------------------------------|
|v       |numpy.ndarray[float64[m, 1]]|A constant reference to the visible configuration.|

### to_dense
Returns the dense matrix representation of the operator. Note that, in general,
the size of the matrix is exponential in the number of quantum
numbers, and this operation should thus only be performed for
low-dimensional Hilbert spaces.

This method requires an indexable Hilbert space.



### to_sparse
Returns the sparse matrix representation of the operator. Note that, in general,
the size of the matrix is exponential in the number of quantum
numbers, and this operation should thus only be performed for
low-dimensional Hilbert spaces or sufficiently sparse operators.

This method requires an indexable Hilbert space.



## Properties

|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |netket.hilbert.Hilbert| ``Hilbert`` space of operator.|
