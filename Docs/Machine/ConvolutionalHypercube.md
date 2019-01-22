# ConvolutionalHypercube
A convolutional feedforward layer for hypercubes. This layer works only for the ``Hypercube`` graph defined in ``graph``. This layer implements the standard convolution with periodic boundary conditions.

## Class Constructor
Constructs a new ``ConvolutionalHypercube`` layer.

|   Argument    |   Type   |                                           Description                                            |
|---------------|----------|--------------------------------------------------------------------------------------------------|
|length         |int       |Size of input images.                                                                             |
|n_dim          |int       |Dimension of the input images.                                                                    |
|input_channels |int       |Number of input channels.                                                                         |
|output_channels|int       |Number of output channels.                                                                        |
|stride         |int=1     |Stride distance.                                                                                  |
|kernel_length  |int=2     |Size of the kernels.                                                                              |
|use_bias       |bool=False|If ``True`` then the transformation will include a bias, i.e., the transformation would be affine.|

### Examples
A ``ConvolutionalHypercube`` layer which takes 4 10x10 input images
and gives 8 10x10 output images by convolving with 4x4 kernels:

```python
>>> from netket.layer import ConvolutionalHypercube
>>> l=ConvolutionalHypercube(length=10,n_dim=2,input_channels=4,output_channels=8,kernel_length=4)
>>> print(l.n_par)
520
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

