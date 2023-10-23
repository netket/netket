(netket_models_api)=
# netket.models

```{eval-rst}
.. currentmodule:: netket.models

```

This sub-module contains several pre-built models to be used as
neural quantum states.

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_model
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

The following autoregressive models can be directly sampled using {class}`~netket.sampler.ARDirectSampler`

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_model
   :nosignatures:

   AbstractARNN
   ARNNDense
   ARNNConv1D
   ARNNConv2D
   FastARNNConv1D
   FastARNNConv2D

   LSTMNet
   GRUNet1D
   FastLSTMNet
   FastGRUNet1D
```

The following models are particularly suited for systems with continuous degrees of freedom ({class}`~netket.hilbert.Particle`)

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_model
   :nosignatures:


   Gaussian
   DeepSetRelDistance
```
