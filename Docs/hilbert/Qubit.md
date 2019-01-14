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
Member function generating uniformely distributed local random states.

```


|Argument|Type|                                   Description                                   |
|--------|----|---------------------------------------------------------------------------------|
|state   |    |A reference to a visible configuration, in output this contains the random state.|
|rgen    |    |The random number generator.                                                     |

### Examples
Test that a new random state is a possible state for the hilbert
space.

```python
>>> import netket as nk
>>> nk.hilbert.Boson(n_max=3, graph=nk.graph.Hypercube(length=5, n_dim=1)
>>> rstate = np.zeros(hi.size)
>>> rg = nk.utils.RandomEngine(seed=1234)
>>> hi.random_vals(rstate, rg)
>>> local_states = hi.local_states
>>> print(rstate[0] in local_states)
True


### update_conf
Member function updating a visible configuration using the information on
where the local changes have been done.

Ars:
v: The vector of visible units to be modified.
tochange: A list of which qunatum numbers will be modified.
newconf: Contains the value that those quantum numbers should take.




## Properties
|  Property  |   Type    |                        Description                        |
|------------|-----------|-----------------------------------------------------------|
|is_discrete |bool       | Whether the hilbert space is discrete.                    |
|local_size  |int        | Size of the local hilbert space.                          |
|local_states|list[float]| List of discreet local quantum numbers.                   |
|size        |int        | The number of visible units needed to describe the system.|
