# ImagTimePropagation
Solving for the ground state of the wavefunction using imaginary time propagation.

## Class Constructor
Constructs an ``ImagTimePropagation`` object from a hamiltonian, a stepper,
a time, and an initial state.

|  Argument   |               Type                |                                                                   Description                                                                    |
|-------------|-----------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
|hamiltonian  |netket::AbstractOperator           |The hamiltonian of the system.                                                                                                                    |
|stepper      |netket.dynamics.AbstractTimeStepper|Stepper (i.e. propagator) that transforms the state of the system from one timestep to the next.                                                  |
|t0           |float                              |The initial time.                                                                                                                                 |
|initial_state|numpy.ndarray[complex128[m, 1]]    |The initial state of the system (when propagation begins.)                                                                                        |
|matrix_type  |str='sparse'                       |The type of matrix used for the Hamiltonian when creating the matrix wrapper. The default is `sparse`. The other choices are `dense` and `direct`.|

### Examples
Solving 1D ising model with imagniary time propagation.

```python
>>> import netket as nk
>>> import numpy as np
>>> L = 20
>>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
>>> hilbert = nk.hilbert.Spin(graph, 0.5)
>>> idx = nk.hilbert.HilbertIndex(hilbert)
>>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
>>> stepper = nk.dynamics.create_timestepper(idx.n_states, rel_tol=1e-10, abs_tol=1e-10)
>>> output = nk.output.JsonOutputWriter('test.log', 'test.wf')
>>> psi0 = np.random.rand(idx.n_states)
>>> driver = nk.exact.ImagTimePropagation(hamiltonian, stepper, t0=0, initial_state=psi0)
>>> driver.add_observable(hamiltonian, 'Hamiltonian')
>>> for step in driver.iter(dt=0.05, n_iter=2):
...     obs = driver.get_observable_stats()

```



## Class Methods 
### add_observable
Add an observable quantity, that will be calculated at each
iteration.

| Argument  |          Type          |                                                                   Description                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
|observable |netket::AbstractOperator|The operator form of the observable.                                                                                                             |
|name       |str                     |The name of the observable.                                                                                                                      |
|matrix_type|str='sparse'            |The type of matrix used for the observable when creating the matrix wrapper. The default is `sparse`. The other choices are `dense` and `direct`.|

### get_observable_stats
Calculate and return the value of the operators stored as observables.




### iter
Iterate the optimization of the Vmc wavefunction.

|Argument|       Type       |               Description               |
|--------|------------------|-----------------------------------------|
|dt      |float             |Number of iterations performed at a time.|
|n_iter  |Optional[int]=None|The maximum number of iterations.        |

## Properties

|Property| Type |      Description       |
|--------|------|------------------------|
|t       |double| Time in the simulation.|
