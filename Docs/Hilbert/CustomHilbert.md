# CustomHilbert
A custom hilbert space.
## Constructor
Constructs a new ``CustomHilbert`` given a graph and a list of 
eigenvalues of the states.

|   Field    |       Type       |         Description          |
|------------|------------------|------------------------------|
|graph       |netket.graph.Graph|Graph representation of sites.|
|local_states|List[float]       |Eigenvalues of the states.    |
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


## Properties
|  Property  |   Type    |                        Description                        |
|------------|-----------|-----------------------------------------------------------|
|is_discrete |bool       | Whether the hilbert space is discrete.                    |
|local_size  |int        | Size of the local hilbert space.                          |
|local_states|list[float]| List of discreet local quantum numbers.                   |
|size        |int        | The number of visible units needed to describe the system.|

