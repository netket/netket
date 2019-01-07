# Vmc
Variational Monte Carlo schemes to learn the ground state using stochastic reconfiguration and gradient descent optimizers.
## Constructor
Constructs a ``VariationalMonteCarlo`` object given a hamiltonian, 
sampler, optimizer, and the number of samples.

|          Field          |   Type   |                                Description                                |
|-------------------------|----------|---------------------------------------------------------------------------|
|hamiltonian              |          |The hamiltonian of the system.                                             |
|sampler                  |          |The sampler object to generate local exchanges.                            |
|optimizer                |          |The optimizer object that determines how the VMC wavefunction is optimized.|
|n_samples                |int       |The total number of samples.                                               |
|discarded_samples        |int=-1    |The number of samples discarded. Default is -1.                            |
|discarded_samples_on_init|int=0     |The number of samples discarded upon initialization. The default is 0.     |
|method                   |str='Sr'  |The solver method. The default is `Sr` (stochastic reconfiguration).       |
|diag_shift               |float=0.01|The diagonal shift. The default is 0.01.                                   |
|rescale_shift            |bool=False|Whether to rescale the variational parameters. The default is false.       |
|use_iterative            |bool=False|Whether to solver iteratively. The default is false.                       |
|use_cholesky             |bool=True |Whether to use cholesky decomposition. The default is true.                |
### Examples
Optimizing a 1D wavefunction with Variational Mante Carlo.

```python
>>> import netket as nk
>>> from mpi4py import MPI
>>> SEED = 3141592
>>> g = nk.graph.Hypercube(length=8, n_dim=1)
>>> hi = nk.hilbert.Spin(s=0.5, graph=g)
>>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
>>> ma.init_random_parameters(seed=SEED, sigma=0.01)
>>> ha = nk.operator.Ising(hi, h=1.0)
>>> sa = nk.sampler.MetropolisLocal(machine=ma)
>>> sa.seed(SEED)
>>> op = nk.optimizer.Sgd(learning_rate=0.1)
>>> vmc = nk.variational.Vmc(hamiltonian=ha,sampler=sa,optimizer=op,n_samples=500)
>>> print(vmc.machine.n_visible)
8

```



## Properties
|Property|Type|Description|
|--------|----|-----------|
|machine |    |           |

