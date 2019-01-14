# Ising
An Ising Hamiltonian operator.

## Class Constructor
Constructs a new ``Ising`` given a hilbert space, a transverse field,
and (if specified) a coupling constant.

|Argument|         Type         |                 Description                 |
|--------|----------------------|---------------------------------------------|
|hilbert |netket.hilbert.Hilbert|Hilbert space the operator acts on.          |
|h       |float                 |The strength of the transverse field.        |
|J       |float=1.0             |The strength of the coupling. Default is 1.0.|

### Examples
Constructs an ``Ising`` operator for a 1D system.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
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

|Argument|Type|                   Description                    |
|--------|----|--------------------------------------------------|
|v       |    |A constant reference to the visible configuration.|

## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |netket.hilbert.Hilbert| ``Hilbert`` space of operator.|
