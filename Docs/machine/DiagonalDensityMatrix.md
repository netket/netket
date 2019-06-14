# DiagonalDensityMatrix
A Machine sampling the diagonal of a density matrix.

## Class Constructor
Constructs a new ``DiagonalDensityMatrix`` machine sampling the
diagonal of the provided density matrix.

|Argument|                 Type                 |    Description    |
|--------|--------------------------------------|-------------------|
|dm      |netket._C_netket.machine.DensityMatrix|the density matrix.|

## Class Methods 
### der_log
Member function to obtain the derivatives of log value of
machine given an input wrt the machine's parameters.

|Argument|            Type            |      Description       |
|--------|----------------------------|------------------------|
|v       |numpy.ndarray[float64[m, 1]]|Input vector to machine.|

### init_random_parameters
Member function to initialise machine parameters.

|Argument|  Type   |                               Description                                |
|--------|---------|--------------------------------------------------------------------------|
|seed    |int=1234 |The random number generator seed.                                         |
|sigma   |float=0.1|Standard deviation of normal distribution from which parameters are drawn.|

### load
Member function to load machine parameters from a json file.

|Argument|Type|             Description             |
|--------|----|-------------------------------------|
|filename|str |name of file to load parameters from.|

### log_norm
Returns the log of the L2 norm of the wave-function.
This operation is a brute-force calculation, and should thus
only be performed for low-dimensional Hilbert spaces.

This method requires an indexable Hilbert space.



### log_val
Member function to obtain log value of machine given an input
vector.

|Argument|            Type            |      Description       |
|--------|----------------------------|------------------------|
|v       |numpy.ndarray[float64[m, 1]]|Input vector to machine.|

### log_val_diff
Member function to obtain difference in log value of machine
given an input and a change to the input.

|Argument|            Type            |                                 Description                                 |
|--------|----------------------------|-----------------------------------------------------------------------------|
|v       |numpy.ndarray[float64[m, 1]]|Input vector to machine.                                                     |
|tochange|List[List[int]]             |list containing the indices of the input to be changed                       |
|newconf |List[List[float]]           |list containing the new (changed) values at the indices specified in tochange|

### save
Member function to save the machine parameters.

|Argument|Type|            Description            |
|--------|----|-----------------------------------|
|filename|str |name of file to save parameters to.|

### to_array
Returns a numpy array representation of the machine.
The returned array is normalized to 1 in L2 norm.
Note that, in general, the size of the array is exponential
in the number of quantum numbers, and this operation should thus
only be performed for low-dimensional Hilbert spaces.

This method requires an indexable Hilbert space.



## Properties

|   Property   |         Type         |                                                   Description                                                    |
|--------------|----------------------|------------------------------------------------------------------------------------------------------------------|
|hilbert       |netket.hilbert.Hilbert| The hilbert space object of the system.                                                                          |
|is_holomorphic|bool                  | Whether the given wave-function is a holomorphic function of             its parameters                          |
|n_par         |int                   | The number of parameters in the machine.                                                                         |
|n_visible     |int                   | The number of inputs into the machine aka visible units in             the case of Restricted Boltzmann Machines.|
|parameters    |list                  | List containing the parameters within the layer.             Read and write                                      |
