(api-experimental)=

# Experimental API

```{eval-rst}
.. currentmodule:: netket.experimental
```

In this page we present some experimental interfaces of NetKet.
Those are not guaranteed to be API-stable, and might change without notice (other than the
changelog) among minor releases.
The {ref}`netket.experimental` modules mirrors the standard {ref}`netket` module structure,
and we suggest to use it as follows:

```python
import netket as nk
import netket.experimental as nkx
```

(experimental-sampler-api)=

## Samplers

This module contains the Metropolis Parallel Tempered sampler.
This sampler is experimental because we believe it to be correct, but our tests
fail. We believe it to be a false negative: possibly the implementation of the
sampler is correct, but the test is too tight.
Until we will have verified this hypotesis and updated the tests in order not
to fail, we provide the current implementation as-is, in the hope that some
contributor might take up that work.

The other experimental sampler is MetropolisSamplerPmap, which makes use of {ref}`jax.pmap`
to use different GPUs/CPUs without having to use MPI. It should scale much better over
several CPUs, but you have to start jax with a specific environment variable.

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/samplers
   :template: class
   :nosignatures:

   netket.experimental.sampler.MetropolisPtSampler
   netket.experimental.sampler.MetropolisLocalPt
   netket.experimental.sampler.MetropolisExchangePt

   netket.experimental.sampler.MetropolisSamplerPmap
```

(experimental-variational-api)=

## Variational State Interface

```{eval-rst}
.. currentmodule:: netket
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/experimental/variational
  :nosignatures:

  netket.experimental.vqs.variables_from_file
  netket.experimental.vqs.variables_from_tar

```

## Time Evolution Driver

```{eval-rst}
.. currentmodule:: netket
```

```{eval-rst}
.. autosummary::
  :toctree: _generated/experimental/dynamics
  :nosignatures:

  netket.experimental.driver.TDVP

```

## ODE Integrators

This is a collection of ODE integrators that can be used with the TDVP
driver above.

```{eval-rst}
.. currentmodule:: netket
```

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/dynamics
   :nosignatures:

  netket.experimental.dynamics.Euler
  netket.experimental.dynamics.Heun
  netket.experimental.dynamics.Midpoint
  netket.experimental.dynamics.RK12
  netket.experimental.dynamics.RK23
  netket.experimental.dynamics.RK4
  netket.experimental.dynamics.RK45
```

## Fermions

This module contains hilbert space and operator implementations of fermions in second quantization.
It is experimental until it has been thoroughly tested by the community, meaning feedback is welcome.

```{eval-rst}
.. currentmodule:: netket
```

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/hilbert
   :template: class
   :nosignatures:

   netket.experimental.hilbert.SpinOrbitalFermions
```

```{eval-rst}
.. autosummary::
   :toctree: _generated/experimental/operator
   :template: class
   :nosignatures:

   netket.experimental.operator.FermionOperator2nd
   netket.experimental.operator.fermion.create
   netket.experimental.operator.fermion.destroy
   netket.experimental.operator.fermion.number
```