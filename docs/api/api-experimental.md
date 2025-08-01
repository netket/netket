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

   driver.VMC_SR
```

Currently NetKet offers an experimental driver implementing Stochastic Reconfiguration to minimize the infidelity between two quantum states (possibly with an operator in the middle, see https://quantum-journal.org/papers/q-2023-10-10-1131 and https://quantum-journal.org/papers/q-2025-07-22-1803/ for references) with the kernel trick or minSR formulation. This is slightly more limited in features than the standard Stochastic Reconfiguration implementation of {class}`netket.drivers.VMC`, but can scale to millions of parameters

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/driver
   :template: class
   :nosignatures:

   driver.Infidelity_SR
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

(experimental-fermions-api)=
## Fermions and PyScf

This modules contains operators for particle-number conserving fermionic operators as well as utility functions that are used to create hamiltonians directly from PyScf molecules.

Previously we also had the remaining Fermionic functionality in the experimental namespace, but in May 2024 it was stabilised and moved to the main netket namespace.

```{eval-rst}
.. currentmodule:: netket.experimental
```

```{eval-rst}
.. autosummary::
    :toctree: _generated/operator
    :template: class
    :nosignatures:

    operator.ParticleNumberConservingFermioperator2nd
    operator.ParticleNumberAndSpinConservingFermioperator2nd
    operator.FermiHubbardJax

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
   observable.InfidelityOperator
```