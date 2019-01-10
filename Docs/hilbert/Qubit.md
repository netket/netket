# Qubit
Hilbert space composed of qubits.

## Class Constructor
Constructs a new ``Qubit`` given a graph.

|Argument|       Type       |         Description          |
|--------|------------------|------------------------------|
|graph   |netket.graph.Graph|Graph representation of sites.|

### Examples
Simple qubit hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import Qubit
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = Qubit(graph=g)
>>> print(hi.size)
100

```



## Class Methods 
### random_vals
### update_conf
## Properties
|  Property  |   Type    |                        Description                        |
|------------|-----------|-----------------------------------------------------------|
|is_discrete |bool       | Whether the hilbert space is discrete.                    |
|local_size  |int        | Size of the local hilbert space.                          |
|local_states|list[float]| List of discreet local quantum numbers.                   |
|size        |int        | The number of visible units needed to describe the system.|
