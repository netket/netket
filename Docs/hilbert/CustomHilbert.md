# CustomHilbert
A custom hilbert space.

## Class Constructor
Constructs a new ``CustomHilbert`` given a graph and a list of
eigenvalues of the states.

|  Argument  |            Type            |         Description          |
|------------|----------------------------|------------------------------|
|graph       |netket._C_netket.graph.Graph|Graph representation of sites.|
|local_states|List[float]                 |Eigenvalues of the states.    |

### Examples
Simple custom hilbert space.

```python
>>> from netket.graph import Hypercube
>>> from netket.hilbert import CustomHilbert
>>> g = Hypercube(length=10,n_dim=2,pbc=True)
>>> hi = CustomHilbert(graph=g, local_states=[-1232, 132, 0])
>>> print(hi.size)
100

```



## Class Methods 
### number_to_state
Returns the visible configuration corresponding to the i-th basis state
for input i. Throws an exception iff the space is not indexable.



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



### state_to_number
Returns index of the given many-body configuration.
Throws an exception iff the space is not indexable.


### states
Returns an iterator over all valid configurations of the Hilbert space.
Throws an exception iff the space is not indexable.


### update_conf
Member function updating a visible configuration using the information on
where the local changes have been done.

|Argument |                    Type                     |                       Description                        |
|---------|---------------------------------------------|----------------------------------------------------------|
|v        |numpy.ndarray[float64[m, 1], flags.writeable]|The vector of visible units to be modified.               |
|to_change|numpy.ndarray[int32]                         |A list of which quantum numbers will be modified.         |
|new_conf |numpy.ndarray[float64]                       |Contains the value that those quantum numbers should take.|

## Properties

|  Property  |                                                                            Type                                                                            |                                                      Description                                                       |
|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------|
|graph       |netket.graph.Graph                                                                                                                                          | The Graph used to construct this Hilbert space.                                                                        |
|index       |        HilbertIndex                                                                                                                                        | An object containing information on the states of an                indexable Hilbert space                            |
|is_discrete |bool                                                                                                                                                        | Whether the hilbert space is discrete.                                                                                 |
|is_indexable|        We call a Hilbert space indexable if and only if the total Hilbert space        dimension can be represented by an index of type int.        Returns|            bool: Whether the Hilbert space is indexable.                                                               |
|local_size  |int                                                                                                                                                         | Size of the local hilbert space.                                                                                       |
|local_states|list[float]                                                                                                                                                 | List of discreet local quantum numbers.                                                                                |
|n_states    |int                                                                                                                                                         | The total dimension of the many-body Hilbert space.                 Throws an exception iff the space is not indexable.|
|size        |int                                                                                                                                                         | The number of visible units needed to describe the system.                                                             |
