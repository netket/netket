# Ising
An Ising Hamiltonian operator.
## Constructor
Constructs a new ``Ising`` given a hilbert space, a transverse field,
and (if specified) a coupling constant.

| Field |         Type         |                 Description                 |
|-------|----------------------|---------------------------------------------|
|hilbert|netket.hilbert.Hilbert|Hilbert space the operator acts on.          |
|h      |float                 |The strength of the transverse field.        |
|J      |float=1.0             |The strength of the coupling. Default is 1.0.|
### Examples
Constructs an ``Ising`` operator for a 1D system.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> op = nk.operator.Ising(h=1.321, hilbert=hi, J=0.5)
# Transverse-Field Ising model created
# h = 1.321
# J = 0.5
```


## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |const AbstractHilbert&| ``Hilbert`` space of operator.|

