# Supervised
Supervised learning scheme to learn data, i.e. the given state, by stochastic gradient descent with log overlap loss or MSE loss.

## Class Constructor
Construct a Supervised object given a machine, an optimizer, batch size and
data, including samples and targets.

|  Argument   |                Type                 |                                                                                    Description                                                                                     |
|-------------|-------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|machine      |netket._C_netket.machine.Machine     |The machine representing the wave function.                                                                                                                                         |
|optimizer    |netket._C_netket.optimizer.Optimizer |The optimizer object that determines how the SGD optimization.                                                                                                                      |
|batch_size   |int                                  |The batch size used in SGD.                                                                                                                                                         |
|samples      |List[numpy.ndarray[float64[m, 1]]]   |The input data, i.e. many-body basis.                                                                                                                                               |
|targets      |List[numpy.ndarray[complex128[m, 1]]]|The output label, i.e. amplitude of the corresponding basis.                                                                                                                        |
|method       |str='Gd'                             |The chosen method to learn the parameters of the wave-function. Possible choices are `Gd` (Regular Gradient descent), and `Sr` (Stochastic reconfiguration a.k.a. natural gradient).|
|diag_shift   |float=0.01                           |The regularization parameter in stochastic reconfiguration. The default is 0.01.                                                                                                    |
|use_iterative|bool=False                           |Whether to use the iterative solver in the Sr method (this is extremely useful when the number of parameters to optimize is very large). The default is false.                      |
|use_cholesky |bool=True                            |Whether to use cholesky decomposition. The default is true.                                                                                                                         |

## Class Methods 
### advance
Run one iteration of supervised learning. This should be helpful for testing and
having self-defined control sequence in python.

|  Argument   |      Type       |                         Description                         |
|-------------|-----------------|-------------------------------------------------------------|
|loss_function|str='Overlap_phi'|The loss function choosing for learning, Default: Overlap_phi|

### run
Run supervised learning.

|    Argument     |      Type       |                         Description                         |
|-----------------|-----------------|-------------------------------------------------------------|
|n_iter           |int              |The number of iterations for running.                        |
|loss_function    |str='Overlap_phi'|The loss function choosing for learning, Default: Overlap_phi|
|output_prefix    |str='output'     |The output file name, without extension.                     |
|save_params_every|int=50           |Frequency to dump wavefunction parameters. The default is 50.|

## Properties

|    Property    | Type |                  Description                   |
|----------------|------|------------------------------------------------|
|loss_log_overlap|double| The current negative log fidelity.             |
|loss_mse        |double| The mean square error of amplitudes.           |
|loss_mse_log    |double| The mean square error of the log of amplitudes.|
