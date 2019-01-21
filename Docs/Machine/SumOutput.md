# SumOutput
A feedforward layer which sums the inputs to give a single output.

## Class Constructor
Constructs a new ``SumOutput`` layer.

| Argument |Type| Description  |
|----------|----|--------------|
|input_size|int |Size of input.|

### Examples
A ``SumOutput`` layer which takes 10-dimensional inputs:

```python
>>> from netket.layer import SumOutput
>>> l=SumOutput(input_size=10)
```



## Class Methods 
### init_random_parameters
Member function to initialise layer parameters.

|Argument|  Type   |                               Description                                |
|--------|---------|--------------------------------------------------------------------------|
|seed    |int=1234 |The random number generator seed.                                         |
|sigma   |float=0.1|Standard deviation of normal distribution from which parameters are drawn.|

## Properties
| Property |Type|                                    Description                                    |
|----------|----|-----------------------------------------------------------------------------------|
|n_input   |int | The number of inputs into the layer.                                              |
|n_output  |int | The number of outputs from the layer.                                             |
|n_par     |int | The number parameters within the layer.                                           |
|parameters|list| List containing the parameters within the layer.             Readable and writable|

