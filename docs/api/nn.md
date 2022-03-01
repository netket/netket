(netket_nn_api)=
# netket.nn

```{eval-rst}
.. currentmodule:: netket.nn

```

This sub-module wraps and re-exports [flax.nn](https://flax.readthedocs.io/en/latest/flax.linen.html).
Read more about the design goal of this module in their [README](https://github.com/google/flax/blob/master/flax/linen/README.md)


## Linear Modules

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   Dense
   DenseGeneral
   DenseSymm
   DenseEquivariant
   Conv
   Embed

   MaskedDense1D
   MaskedConv1D
   MaskedConv2D

```

## Activation functions

```{eval-rst}
.. autosummary::
    :toctree: _generated/nn

    celu
    elu
    gelu
    glu
    log_sigmoid
    log_softmax
    relu
    sigmoid
    soft_sign
    softmax
    softplus
    swish
    log_cosh
    reim_relu
    reim_selu

```
