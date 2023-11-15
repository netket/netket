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
   :template: flax_module_or_default
   :nosignatures:

   DenseSymm
   DenseEquivariant
```

The following modules can be used in autoregressive neural networks, see {class}`~netket.models.AbstractARNN`

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn
   :template: flax_module_or_default
   :nosignatures:

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
   :template: flax_module_or_default
   :nosignatures:

    blocks.MLP
    blocks.DeepSetMLP
    blocks.SymmExpSum

```

## Experimental

### Recurrent Neural Network cells

The following are RNN layers (in flax those would be called a RNN), which can be stacked within a {class}`netket.experimental.models.RNN`.


```{eval-rst}
.. currentmodule:: netket.experimental.nn

```

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn/rnn
   :template: flax_module_or_default
   :nosignatures:

   rnn.RNNLayer
   rnn.FastRNNLayer
```

The following are recurrent cells that can be used with {class}`netket.experimental.nn.rnn.RNNLayer`.

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn/rnn
   :template: flax_module_or_default
   :nosignatures:

   rnn.RNNCell
   rnn.LSTMCell
   rnn.GRU1DCell
```

The following are utility functions to build up custom autoregressive orderings.

```{eval-rst}
.. autosummary::
   :toctree: _generated/nn/rnn
   :template: flax_module_or_default
   :nosignatures:

   rnn.check_reorder_idx
   rnn.ensure_prev_neighbors
   rnn.get_snake_inv_reorder_idx
```


