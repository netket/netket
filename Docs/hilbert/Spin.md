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
Member function generating uniformely distributed local random states.

|Argument|                   Type                   |                                   Description                                   |
|--------|------------------------------------------|---------------------------------------------------------------------------------|
|state   |numpy.ndarray[float64[m                   |A reference to a visible configuration, in output this contains the random state.|
|rgen    |std::mersenne_twister_engine<unsigned long|The random number generator.                                                     |

### Examples
Test that a new random state is a possible state for the hilbert
space.

```python
>>> import netket as nk
>>> import numpy as np
>>> hi = nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1))
>>> rstate = np.zeros(hi.size)
>>> rg = nk.utils.RandomEngine(seed=1234)
>>> hi.random_vals(rstate, rg)
>>> local_states = hi.local_states
>>> print(rstate[0] in local_states)
True

```



### update_conf
Member function updating a visible configuration using the information on
where the local changes have been done.

|Argument |         Type          |                       Description                        |
|---------|-----------------------|----------------------------------------------------------|
|v        |numpy.ndarray[float64[m|The vector of visible units to be modified.               |
|to_change|List[int]              |A list of which qunatum numbers will be modified.         |
|new_conf |List[float]            |Contains the value that those quantum numbers should take.|

## Properties
|  Property  |   Type    |                        Description                        |
|------------|-----------|-----------------------------------------------------------|
|is_discrete |bool       | Whether the hilbert space is discrete.                    |
|local_size  |int        | Size of the local hilbert space.                          |
|local_states|list[float]| List of discreet local quantum numbers.                   |
|size        |int        | The number of visible units needed to describe the system.|
