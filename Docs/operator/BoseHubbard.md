# BoseHubbard
A Bose Hubbard model Hamiltonian operator.

## Class Constructor
Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
interaction strength. The chemical potential and the hopping term can
be specified as well.

|Argument|         Type         |            Description            |
|--------|----------------------|-----------------------------------|
|hilbert |netket.hilbert.Hilbert|Hilbert space the operator acts on.|
|U       |float                 |The Hubbard interaction term.      |
|V       |float=0.0             |The hopping term.                  |
|mu      |float=0.0             |The chemical potential.            |

### Examples
Constructs a ``BoseHubbard`` operator for a 2D system.

```python
>>> from mpi4py import MPI
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

|Argument|         Type          |                   Description                    |
|--------|-----------------------|--------------------------------------------------|
|v       |numpy.ndarray[float64[m|A constant reference to the visible configuration.|

## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |netket.hilbert.Hilbert| ``Hilbert`` space of operator.|
