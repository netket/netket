# Spin
Hilbert space composed of spin states.

## Class Constructor [1]
Constructs a new ``Spin`` given a graph and the value of each spin.

|Argument|       Type       |                    Description                    |
|--------|------------------|---------------------------------------------------|
|graph   |netket.graph.Graph|Graph representation of sites.                     |
|s       |float             |Spin at each site. Must be integer or half-integer.|

### Examples
Simple spin hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import Spin
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = Spin(graph=g, s=0.5)
>>> print(hi.size)
100

```


## Class Constructor [2]
Constructs a new ``Spin`` given a graph and the value of each spin.

|Argument|       Type       |                     Description                     |
|--------|------------------|-----------------------------------------------------|
|graph   |netket.graph.Graph|Graph representation of sites.                       |
|s       |float             |Spin at each site. Must be integer or half-integer.  |
|total_sz|float             |Constrain total spin of system to a particular value.|

### Examples
Simple spin hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import Spin
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = Spin(graph=g, s=0.5, total_sz=0)
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
