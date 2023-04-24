(netket_sampler_api)=
# netket.sampler

```{eval-rst}
.. currentmodule:: netket.sampler

```


```{eval-rst}
.. inheritance-diagram:: netket.sampler netket.sampler.rules
   :top-classes: netket.sampler.Sampler
   :parts: 1

```

## Abstract Classes

Below you find a list of all abstract classes defined in this module, from which you can inherit if you want to define new hilbert spaces.

```{eval-rst}
.. currentmodule:: netket.sampler

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   Sampler
   SamplerState
   MetropolisSamplerState
   MetropolisRule
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
   ARDirectSampler

```

```{eval-rst}
.. currentmodule:: netket

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   experimental.sampler.MetropolisPtSampler

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

```{eval-rst}
.. currentmodule:: netket

.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   experimental.sampler.MetropolisLocalPt
   experimental.sampler.MetropolisExchangePt
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

```

There are also a few additional rules that can be used to compose other rules together.
```
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