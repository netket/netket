# Vmc
Variational Monte Carlo schemes to learn the ground state using stochastic reconfiguration and gradient descent optimizers.

## Class Constructor
Constructs a ``VariationalMonteCarlo`` object given a hamiltonian,
sampler, optimizer, and the number of samples.

|        Argument         |          Type           |                                                                                    Description                                                                                     |
|-------------------------|-------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|hamiltonian              |netket::AbstractOperator |The hamiltonian of the system.                                                                                                                                                      |
|sampler                  |netket::AbstractSampler  |The sampler object to generate local exchanges.                                                                                                                                     |
|optimizer                |netket::AbstractOptimizer|The optimizer object that determines how the VMC wavefunction is optimized.                                                                                                         |
|n_samples                |int                      |Number of Markov Chain Monte Carlo sweeps to be performed at each step of the optimization.                                                                                         |
|discarded_samples        |int=-1                   |Number of sweeps to be discarded at the beginning of the sampling, at each step of the optimization. Default is -1.                                                                 |
|discarded_samples_on_init|int=0                    |Number of sweeps to be discarded in the first step of optimization, at the beginning of the sampling. The default is 0.                                                             |
|target                   |str='energy'             |The chosen target to minimize. Possible choices are `energy`, and `variance`.                                                                                                       |
|method                   |str='Sr'                 |The chosen method to learn the parameters of the wave-function. Possible choices are `Gd` (Regular Gradient descent), and `Sr` (Stochastic reconfiguration a.k.a. natural gradient).|
|diag_shift               |float=0.01               |The regularization parameter in stochastic reconfiguration. The default is 0.01.                                                                                                    |
|use_iterative            |bool=False               |Whether to use the iterative solver in the Sr method (this is extremely useful when the number of parameters to optimize is very large). The default is false.                      |
|use_cholesky             |bool=True                |Whether to use cholesky decomposition. The default is true.                                                                                                                         |

### Examples
Optimizing a 1D wavefunction with Variational Mante Carlo.

```python
>>> import netket as nk
>>> SEED = 3141592
>>> g = nk.graph.Hypercube(length=8, n_dim=1)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
>>> ma.init_random_parameters(seed=SEED, sigma=0.01)
>>> ha = nk.operator.Ising(hi, h=1.0)
>>> sa = nk.sampler.MetropolisLocal(machine=ma)
>>> sa.seed(SEED)
>>> op = nk.optimizer.Sgd(learning_rate=0.1)
>>> vmc = nk.variational.Vmc(hamiltonian=ha, sampler=sa,
... optimizer=op, n_samples=500)
>>> print(vmc.machine.n_visible)
8

```




## Class Methods 
### add_observable
Add an observable quantity, that will be calculated at each
iteration.

|Argument|          Type          |            Description             |
|--------|------------------------|------------------------------------|
|ob      |netket::AbstractOperator|The operator form of the observable.|
|ob_name |str                     |The name of the observable.         |

### advance
Perform one or several iteration steps of the VMC calculation. In each step,
energy and gradient will be estimated via VMC and subsequently, the variational
parameters will be updated according to the configured method.

|Argument|Type |          Description          |
|--------|-----|-------------------------------|
|steps   |int=1|Number of VMC steps to perform.|

### get_observable_stats
Calculate and return the value of the operators stored as observables.




### iter
Returns a generator which advances the VMC optimization, yielding
after every step_size steps up to n_iter.


|Argument |  Type  |                  Description                  |
|---------|--------|-----------------------------------------------|
|n_iter   |int=None|The number of steps or None, for no limit.     |
|step_size|int=1   |The number of steps the simulation is advanced.|

### reset
### run
Optimize the Vmc wavefunction.

|    Argument     |       Type       |                         Description                         |
|-----------------|------------------|-------------------------------------------------------------|
|output_prefix    |str               |The output file name, without extension.                     |
|n_iter           |Optional[int]=None|The maximum number of iterations.                            |
|step_size        |int=1             |Number of iterations performed at a time. Default is 1.      |
|save_params_every|int=50            |Frequency to dump wavefunction parameters. The default is 50.|

### Examples
Running a simple Vmc calculation.


```python
>>> import netket as nk
>>> SEED = 3141592
>>> g = nk.graph.Hypercube(length=8, n_dim=1)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
>>> ma.init_random_parameters(seed=SEED, sigma=0.01)
>>> ha = nk.operator.Ising(hi, h=1.0)
>>> sa = nk.sampler.MetropolisLocal(machine=ma)
>>> sa.seed(SEED)
>>> op = nk.optimizer.Sgd(learning_rate=0.1)
>>> vmc = nk.variational.Vmc(hamiltonian=ha, sampler=sa,
... optimizer=op, n_samples=500)
>>> vmc.run(output_prefix='test', n_iter=1)


```




## Properties

|Property|         Type         |                 Description                  |
|--------|----------------------|----------------------------------------------|
|machine |netket.machine.Machine| The machine used to express the wavefunction.|
|vmc_data|                      |                                              |
