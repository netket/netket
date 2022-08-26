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
   AbstractARNN
   ARNNDense
   ARNNConv1D
   ARNNConv2D
   FastARNNConv1D
   FastARNNConv2D
   GRUNet1D
   LSTMNet1D
   LSTMNet2D
   FastGRUNet1D
   FastLSTMNet1D
   FastLSTMNet2D
   DeepSetMLP
   MLP

```

The following models are particularly suited for systems with continuous degrees of freedom (:class:`nk.hilbert.Particle`)

```{eval-rst}
.. autosummary::
   :toctree: _generated/models
   :template: flax_model
   :nosignatures:


   Gaussian
   DeepSetRelDistance
```