# BoseHubbard
A Bose Hubbard model Hamiltonian operator.

## Class Constructor
Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
interaction strength. The chemical potential and the hopping term can
be specified as well.

|Argument|              Type              |            Description            |
|--------|--------------------------------|-----------------------------------|
|hilbert |netket._C_netket.hilbert.Hilbert|Hilbert space the operator acts on.|
|U       |float                           |The Hubbard interaction term.      |
|V       |float=0.0                       |The hopping term.                  |
|mu      |float=0.0                       |The chemical potential.            |

### Examples
Constructs a ``BoseHubbard`` operator for a 2D system.

```python
>>> import netket as nk
>>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
>>> hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
>>> op = nk.operator.BoseHubbard(U=4.0, hilbert=hi)
>>> print(op.hilbert.size)
9

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
