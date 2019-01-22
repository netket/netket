# FullyConnected
A fully connected feedforward layer. This layer implements the transformation from a m-dimensional input vector $$ \boldsymbol{v}_n $$ to a n-dimensional output vector $$ \boldsymbol{v}_{n+1} $$: $$ \boldsymbol{v}_n \rightarrow \boldsymbol{v}_{n+1} = g_{n}(\boldsymbol{W}{n}\boldsymbol{v}{n} + \boldsymbol{b}_{n} ) $$ where $$ \boldsymbol{W}{n} $$ is a m by n weights matrix and $$ \boldsymbol{b}_{n} $$ is a n-dimensional bias vector.

## Class Constructor
Constructs a new ``FullyConnected`` layer given input and output
sizes.

| Argument  |   Type   |                                           Description                                            |
|-----------|----------|--------------------------------------------------------------------------------------------------|
|input_size |int       |Size of input to the layer (Length of input vector).                                              |
|output_size|int       |Size of output from the layer (Length of output vector).                                          |
|use_bias   |bool=False|If ``True`` then the transformation will include a bias, i.e., the transformation would be affine.|

### Examples
A ``FullyConnected`` layer which takes 10-dimensional inputs
and gives a 20-dimensional output:

```python
>>> from netket.layer import FullyConnected
>>> l=FullyConnected(input_size=10,output_size=20,use_bias=True)
>>> print(l.n_par)
220

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

