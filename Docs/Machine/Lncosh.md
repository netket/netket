# Lncosh
An activation layer which applies Lncosh to each input.

## Class Constructor
Constructs a new ``Lncosh`` activation layer.

| Argument |Type| Description  |
|----------|----|--------------|
|input_size|int |Size of input.|

### Examples
A ``Lncosh`` activation layer which applies the Lncosh function
coefficient-wise to a 10-dimensional input:

```python
>>> from netket.layer import Lncosh
>>> l=Lncosh(input_size=10)
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

