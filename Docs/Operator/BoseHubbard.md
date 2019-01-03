# BoseHubbard
A Bose Hubbard model Hamiltonian operator.
## Constructor
Constructs a new ``BoseHubbard`` given a hilbert space and a Hubbard
interaction strength. The chemical potential and the hopping term can
be specified as well.

| Field |         Type         |            Description            |
|-------|----------------------|-----------------------------------|
|hilbert|netket.hilbert.Hilbert|Hilbert space the operator acts on.|
|U      |float                 |The Hubbard interaction term.      |
|V      |float=0.0             |The hopping term.                  |
|mu     |float=0.0             |The chemical potential.            |
### Examples
Constructs a ``BoseHubbard`` operator for a 2D system.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> g = nk.graph.Hypercube(length=3, n_dim=2, pbc=True)
>>> hi = nk.hilbert.Boson(n_max=3, n_bosons=6, graph=g)
>>> op = nk.operator.BoseHubbard(U=4.0, hilbert=hi)
# Bose Hubbard model created
# U= 4
# V= 0
# mu= 0
# Nmax= 3
```


## Properties
|Property|         Type         |          Description          |
|--------|----------------------|-------------------------------|
|hilbert |const AbstractHilbert&| ``Hilbert`` space of operator.|

