# Heisenberg
A Heisenberg Hamiltonian operator.
## Constructor
Constructs a new ``Heisenberg`` given a hilbert space.

| Field |         Type         |            Description            |
|-------|----------------------|-----------------------------------|
|hilbert|netket.hilbert.Hilbert|Hilbert space the operator acts on.|
### Examples
Constructs a ``Heisenberg`` operator for a 1D system.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
>>> hi = nk.hilbert.Spin(s=0.5, total_sz=0, graph=g)
>>> op = nk.operator.Heisenberg(hilbert=hi)
# Heisenberg model created
```


## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |netket.hilbert.Hilbert| ``Hilbert`` space of operator.|

