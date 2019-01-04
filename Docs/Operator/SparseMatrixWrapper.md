# SparseMatrixWrapper
This class stores the matrix elements of a given Operator as an Eigen sparse matrix.
## Constructor
Constructs a sparse matrix wrapper from an operator. Matrix elements are
stored as a sparse Eigen matrix.

| Field  |     Type      |               Description                |
|--------|---------------|------------------------------------------|
|operator|netket.Operator|The operator used to construct the matrix.|
### Examples
Printing the dimension of a sparse matrix wrapper.

```python
>>> import netket as nk
>>> from mpi4py import MPI
>>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> op = nk.operator.Ising(h=1.321, hilbert=hi)
# Transverse-Field Ising model created
# h = 1.321
# J = 1
>>> smw = nk.operator.SparseMatrixWrapper(op)
>>> smw.dimension
1048576
```


## Properties
|Property |           Type            |                         Description                         |
|---------|---------------------------|-------------------------------------------------------------|
|_matrix  |Eigen SparseMatrix Complex | The stored matrix.                                          |
|dimension|int                        | The Hilbert space dimension corresponding to the Hamiltonian|

