# DirectMatrixWrapper
This class wraps a given Operator. The
        matrix elements are not stored separately but are computed from
        Operator::FindConn every time Apply is called.

## Class Constructor
Constructs a direct matrix wrapper from an operator. Matrix elements are
calculated when required.

|Argument|     Type      |               Description                |
|--------|---------------|------------------------------------------|
|operator|netket.Operator|The operator used to construct the matrix.|

### Examples
Printing the dimension of a direct matrix wrapper.

```python
>>> import netket as nk
>>> from mpi4py import MPI
>>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> op = nk.operator.Ising(h=1.321, hilbert=hi)
>>> dmw = nk.operator.DirectMatrixWrapper(op)
>>> print(dmw.dimension)
1048576

```




## Class Methods 
### apply
## Properties

|Property |Type|                         Description                         |
|---------|----|-------------------------------------------------------------|
|dimension|int | The Hilbert space dimension corresponding to the Hamiltonian|
