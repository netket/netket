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

|Argument|                                                                               Type                                                                               |                                   Description                                   |
|--------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------|
|state   |numpy.ndarray[float64[m, 1], flags.writeable]                                                                                                                     |A reference to a visible configuration, in output this contains the random state.|
|rgen    |std::mersenne_twister_engine<unsigned long, 32ul, 624ul, 397ul, 31ul, 2567483615ul, 11ul, 4294967295ul, 7ul, 2636928640ul, 15ul, 4022730752ul, 18ul, 1812433253ul>|The random number generator.                                                     |

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

|Argument |                    Type                     |                       Description                        |
|---------|---------------------------------------------|----------------------------------------------------------|
|v        |numpy.ndarray[float64[m, 1], flags.writeable]|The vector of visible units to be modified.               |
|to_change|List[int]                                    |A list of which qunatum numbers will be modified.         |
|new_conf |List[float]                                  |Contains the value that those quantum numbers should take.|

## Properties

|  Property  |                                                                            Type                                                                            |                                        Description                                         |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
|index       |        HilbertIndex                                                                                                                                        | An object containing information on the states of an                indexable Hilbert space|
|is_discrete |bool                                                                                                                                                        | Whether the hilbert space is discrete.                                                     |
|is_indexable|        We call a Hilbert space indexable if and only if the total Hilbert space        dimension can be represented by an index of type int.        Returns|            bool: Whether the Hilbert space is indexable.                                   |
|local_size  |int                                                                                                                                                         | Size of the local hilbert space.                                                           |
|local_states|list[float]                                                                                                                                                 | List of discreet local quantum numbers.                                                    |
|size        |int                                                                                                                                                         | The number of visible units needed to describe the system.                                 |
