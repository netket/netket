(netket_sampler_api)=
# netket.sampler


```{eval-rst}
.. inheritance-diagram:: netket.sampler netket.sampler.rules
   :top-classes: netket.sampler.base.Sampler netket.sampler.rules.base.MetropolisRule
   :parts: 1

```

```{eval-rst}
.. currentmodule:: netket.sampler

```

## Abstract Classes

Below you find a list of all abstract classes defined in this module, from which you can inherit if you want to define new samplers spaces.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   Sampler
   SamplerState
   MetropolisSamplerState
```
## List of Samplers

This is a list of all available samplers.
Please note that samplers with `Numpy` in their name are implemented in
Numpy and not in pure jax, and they will convert from numpy\<->jax at every
sampling step the state.
If you are using GPUs, this conversion can be very costly. On CPUs, while the
conversion is cheap, the dispatch cost of jax is considerate for small systems.

In general those samplers, while they have the same asymptotic cost of Jax samplers,
have a much higher overhead for small to moderate (for GPUs) system sizes.

This is because it is not possible to implement all transition rules in Jax.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   ExactSampler
   MetropolisSampler
   MetropolisSamplerNumpy
   ParallelTemperingSampler
   ARDirectSampler

```

This is a list of shorthands that allow to construct a {class}`~netket.sampler.MetropolisSampler` with a corresponding rule.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers

   MetropolisLocal
   MetropolisExchange
   MetropolisHamiltonian
   MetropolisGaussian
   MetropolisAdjustedLangevin
```

This is an equivalent list of shorthands that allow to construct a {class}`~netket.sampler.ParallelTemperingSampler` with a corresponding rule.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   ParallelTemperingLocal
   ParallelTemperingExchange
   ParallelTemperingHamiltonian
```

The following samplers are for 2nd-quantisation fermionic hilbert spaces ({class}`netket.hilbert.SpinOrbitalFermions`).

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers
   :template: flax_module_or_default
   :nosignatures:


   MetropolisParticleExchange
```

### Transition Rules

Those are the transition rules that can be used with the Metropolis
Sampler. Rules with `Numpy` in their name can only be used with
{class}`netket.sampler.MetropolisSamplerNumpy`.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
  :toctree: _generated/samplers

  rules.MetropolisRule
  rules.LocalRule
  rules.CustomRuleNumpy
  rules.ExchangeRule
  rules.FixedRule
  rules.HamiltonianRule
  rules.HamiltonianRuleNumpy
  rules.GaussianRule
  rules.LangevinRule
  rules.FermionHopRule

```

There are also a few additional rules that can be used to compose other rules together.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
  :toctree: _generated/samplers

  rules.TensorRule
  rules.MultipleRules

```

### Internal State

Those structure hold the state of the sampler.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
  :toctree: _generated/samplers

  SamplerState
  MetropolisSamplerState
```

### Experimental

They are experimental, meaning that we could change them at some point, and we actively seeking for feedback and opinions on their usage and APIs.

```{eval-rst}
.. currentmodule:: netket.experimental.sampler

```

### Particle-specific samplers

The following samplers are for 2nd-quantisation fermionic hilbert spaces ({class}`netket.experimental.hilbert.SpinOrbitalFermions`).

```{eval-rst}
.. autosummary::
   :toctree: _generated/samplers
   :template: flax_module_or_default
   :nosignatures:


   MetropolisFermionHop
```

And the corresponding rules
```{eval-rst}
.. autosummary::
   :toctree: _generated/samplers
   :template: flax_module_or_default
   :nosignatures:


   rules.FermionHopRule
```
