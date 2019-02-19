# ImagTimePropagation
Solving for the ground state of the wavefunction using imaginary time propagation.

## Class Constructor
Constructs an ``ImagTimePropagation`` object from a hamiltonian, a stepper,
a time, and an initial state.

|  Argument   |                                          Type                                           |                                          Description                                           |
|-------------|-----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|
|hamiltonian  |netket::AbstractMatrixWrapper<Eigen::Matrix<std::__1::complex<double>, -1, 1, 0, -1, 1> >|The hamiltonian of the system.                                                                  |
|stepper      |netket.dynamics.AbstractTimeStepper                                                      |Stepper (i.e. propagator) that transforms the state of the system from one timestep to the next.|
|t0           |float                                                                                    |The initial time.                                                                               |
|initial_state|numpy.ndarray[complex128[m, 1]]                                                          |The initial state of the system (when propagation begins.)                                      |

### Examples
Solving 1D ising model with imagniary time propagation.

```python
>>> from mpi4py import MPI
>>> import netket as nk
>>> import numpy as np
>>> L = 20
>>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
>>> hilbert = nk.hilbert.Spin(graph, 0.5)
>>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
>>> mat = nk.operator.wrap_as_matrix(hamiltonian)
>>> stepper = nk.dynamics.create_timestepper(mat.dimension, rel_tol=1e-10, abs_tol=1e-10)
>>> output = nk.output.JsonOutputWriter('test.log', 'test.wf')
>>> psi0 = np.random.rand(mat.dimension)
>>> driver = nk.exact.ImagTimePropagation(mat, stepper, t0=0, initial_state=psi0)
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
|matrix_type|str='Sparse'            |The type of matrix used for the observable when creating the matrix wrapper. The default is `Sparse`. The other choices are `Dense` and `Direct`.|

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
