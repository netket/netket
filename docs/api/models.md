(netket_models_api)=
# netket.models

```{eval-rst}
.. currentmodule:: netket.models

```

This sub-module contains several pre-built models to be used as
neural quantum states.

## Generic models

This section lists some simple variational architectures.

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:

   LogStateVector
   RBM
   RBMModPhase
   RBMMultiVal
   RBMSymm
   Jastrow
   MPSPeriodic
   NDM
   GCNN

   DeepSetMLP
   MLP

```


## Autoregressive models

The following autoregressive models can be directly sampled using {class}`~netket.sampler.ARDirectSampler`.

Those that follow are the abstract classes that can be inherited from in order to build an autoregressive model.

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:

   AbstractARNN
   ARNNSequential
   FastARNNSequential
```

And those are some default implementation of Dense and Convolutional-autoregressive Neural networks. Those are built by using masked dense and masked convolutional layers.

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:

   ARNNDense
   ARNNConv1D
   ARNNConv2D
   FastARNNConv1D
   FastARNNConv2D

```


## Continuous degrees of freedom

The following models are particularly suited for systems with continuous degrees of freedom ({class}`~netket.hilbert.Particle`)

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:


   Gaussian
   DeepSetRelDistance
```


## Experimental models

The following models are experimental, meaning that we could change them at some point, and we actively seeking for feedback and opinions on their usage and APIs.


### Fermionic models

The following models are for 2nd-quantisation fermionic hilbert spaces ({class}`netket.experimental.hilbert.SpinOrbitalFermions`).

```{eval-rst}
.. currentmodule:: netket.experimental.models

```

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:


   Slater2nd
```

### Recurrent Neural Networks (RNN)

```{eval-rst}
.. currentmodule:: netket.experimental.models

```

The following are abstract models for Recurrent neural networks (and their fast versions).

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:


   RNN
   FastRNN
```

The following are concrete, ready to use versions of Recurrent Neural Networks

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_module_or_default
   :nosignatures:

   LSTMNet
   FastLSTMNet
   GRUNet1D
   FastGRUNet1D
```
