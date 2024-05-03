# Experimental API

```{eval-rst}
.. currentmodule:: netket.experimental
```

In this page we present some experimental interfaces of NetKet.
Those are not guaranteed to be API-stable, and might change without notice (other than the
changelog) among minor releases.
The [netket.experimental](api-experimental) module mirrors the standard {ref}`netket <netket_api>` module structure,
and we suggest to use it as follows:

```python
import netket as nk
import netket.experimental as nkx
```

(experimental-drivers-api)=
## Drivers
Currently NetKet offers an experimental driver implementing Stochastic Reconfiguration with the [kernel trick](https://arxiv.org/abs/2310.05715) 
(originally introduced under the name of [minSR by Ao Chen and Markus Heyl](https://arxiv.org/abs/2302.01941)). This is slightly more limited in
features than the standard Stochastic Reconfiguration implementation of {class}`netket.drivers.VMC`, but can scale to millions of parameters

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/driver
   :template: class
   :nosignatures:

   driver.VMC_SRt
```


(experimental-qsr-api)=
## Quantum State Reconstruction
The Quantum State Reconstruction algorithm performs an approximate tomographic reconstruction of measurement data coming from a quantum computer (or similar device) using a Pure or Mixed quantum state.

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/qsr
   :template: class
   :nosignatures:

   QSR
```

(experimental-sampler-api)=
## Samplers

They are experimental, meaning that we could change them at some point, and we actively seeking for feedback and opinions on their usage and APIs.

```{eval-rst}
.. currentmodule:: netket.experimental

```

### Parallel tempering samplers

An experimental sampler is MetropolisSamplerPmap, which makes use of {func}`jax.pmap`
to use different GPUs/CPUs without having to use MPI. It should scale much better over
several CPUs, but you have to start jax with a specific environment variable.


```{eval-rst}
.. autosummary::
   :toctree: _generated/samplers
   :template: class
   :nosignatures:

   sampler.MetropolisSamplerPmap
```

(experimental-fermions-api)=
## Fermions and PyScf

This modules contains some utility functions that are used to create hamiltonians directly from some PyScf molecules. 
Previously we also had several Fermionic functionality in the experimental namespace, but in May 2024 they were stabilised and moved to the main netket namespace.

```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
    :toctree: _generated/operator
    :template: class
    :nosignatures:

    operator.from_pyscf_molecule
    operator.pyscf.TV_from_pyscf_molecule
```

(experimental-logging-api)=
## Logging

```{eval-rst}
.. currentmodule:: netket.experimental

```

This module contains experimental loggers that can be used with the optimization drivers.


```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/logging
   :nosignatures:

   logging.HDF5Log

```

(experimental-variational-api)=
## Variational State Interface

```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/experimental/variational
  :nosignatures:

  vqs.variables_from_file
  vqs.variables_from_tar

```

## Time Evolution Driver


```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/experimental/dynamics
  :nosignatures:

  TDVP
  driver.TDVPSchmitt
```

## ODE Integrators

### Concrete solvers
This is a collection of ODE solvers that can be used with the TDVP driver above.

```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/dynamics
   :nosignatures:

   dynamics.Euler
   dynamics.Heun
   dynamics.Midpoint
   dynamics.RK12
   dynamics.RK23
   dynamics.RK4
   dynamics.RK45
```
The corresponding integrator is then automatically constructed within the TDVP driver. 


### Abstract classes 
Those are the abstract classes you can inherit from to implement your own solver
```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/dynamics
   :template: class
   :nosignatures:

   dynamics.AbstractSolver
   dynamics.AbstractSolverState
```


## Observables
This module contains various observables that can be computed starting from various variational states.

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/observable
   :template: class
   :nosignatures:

   observable.Renyi2EntanglementEntropy
   observable.VarianceObservable
```