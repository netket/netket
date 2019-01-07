# Boson
Hilbert space composed of bosonic states.
## Constructor [1]
Constructs a new ``Boson`` given a graph and maximum occupation number.

|Field|       Type       |         Description          |
|-----|------------------|------------------------------|
|graph|netket.graph.Graph|Graph representation of sites.|
|n_max|int               |Maximum occupation for a site.|
### Examples
Simple boson hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import Boson
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = Boson(graph=g, n_max=4)
>>> print(hi.size)
100

```

## Constructor [2]
Constructs a new ``Boson`` given a graph,  maximum occupation number,
and total number of bosons.

| Field  |       Type       |            Description             |
|--------|------------------|------------------------------------|
|graph   |netket.graph.Graph|Graph representation of sites.      |
|n_max   |int               |Maximum occupation for a site.      |
|n_bosons|int               |Constraint for the number of bosons.|
### Examples
Simple boson hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import Boson
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = Boson(graph=g, n_max=5, n_bosons=11)
>>> print(hi.size)
100

```


## Properties
|  Property  |   Type    |                        Description                        |
|------------|-----------|-----------------------------------------------------------|
|is_discrete |bool       | Whether the hilbert space is discrete.                    |
|local_size  |int        | Size of the local hilbert space.                          |
|local_states|list[float]| List of discreet local quantum numbers.                   |
|size        |int        | The number of visible units needed to describe the system.|

