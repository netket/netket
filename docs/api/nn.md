(netket_nn_api)=
# netket.nn

```{eval-rst}
.. currentmodule:: netket.nn

```

This sub-module extends [flax.linen](https://flax.readthedocs.io/en/latest/flax.linen.html) with layers and tools that are useful to applications in quantum physics.
Read more about the design goal of this module in their [README](https://github.com/google/flax/blob/master/flax/linen/README.md)


## Linear Modules

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   DenseSymm
   DenseEquivariant

   MaskedDense1D
   MaskedConv1D
   MaskedConv2D

   FastMaskedDense1D
   FastMaskedConv1D
   FastMaskedConv2D

```

## Activation functions

```{eval-rst}
.. autosummary::
    :toctree: _generated/nn

    activation.reim
    activation.reim_relu
    activation.reim_selu
    activation.log_cosh
    activation.log_sinh
    activation.log_tanh

```

## Miscellaneous Functions

```{eval-rst}
.. autosummary::
    :toctree: _generated/nn

    binary_encoding
    states_to_numbers
```

## Utility functions

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn

   to_array
   to_matrix

```


## Blocks

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

    blocks.MLP

```
