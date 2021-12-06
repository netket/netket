
.. _api-experimental:

################
Experimental API
################

.. currentmodule:: netket.experimental

In this page we present some experimental interfaces of NetKet.
Those are not guaranteed to be API-stable, and might change without notice (other than the
changelog) among minor releases. 
The :ref:`netket.experimental` modules mirrors the standard :ref:`netket` module structure,
and we suggest to use it as follows:

.. code:: python
	
	import netket as nk
	import netket.experimental as nkx


.. _experimental-sampler-api:

Samplers
~~~~~~~~

This module contains the Metropolis Parallel Tempered sampler.
This sampler is experimental because we believe it to be correct, but our tests
fail. We believe it to be a false negative: possibly the implementation of the
sampler is correct, but the test is too tight. 
Until we will have verified this hypotesis and updated the tests in order not
to fail, we provide the current implementation as-is, in the hope that some
contributor might take up that work.

The other experimental sampler is MetropolisSamplerPmap, which makes use of :ref:`jax.pmap`
to use different GPUs/CPUs without having to use MPI. It should scale much better over
several CPUs, but you have to start jax with a specific environment variable.


.. autosummary::
   :toctree: _generated/experimental/samplers
   :template: class
   :nosignatures:

   netket.experimental.sampler.MetropolisPtSampler
   netket.experimental.sampler.MetropolisLocalPt
   netket.experimental.sampler.MetropolisExchangePt

   netket.experimental.sampler.MetropolisSamplerPmap

.. _experimental-variational-api:

Variational State Interface
---------------------------

.. currentmodule:: netket

.. autosummary::
  :toctree: _generated/experimental/variational
  :nosignatures:

  netket.experimental.vqs.variables_from_file
  netket.experimental.vqs.variables_from_tar


Time Evolution Driver
---------------------

.. currentmodule:: netket

.. autosummary::
  :toctree: _generated/experimental/dynamics
  :nosignatures:

  netket.experimental.driver.TDVP


ODE Integrators
~~~~~~~~~~~~~~~

This is a collection of ODE integrators that can be used with the TDVP 
driver above.

.. currentmodule:: netket

.. autosummary::
  :toctree: _generated/experimental/dynamics
  :nosignatures:


  nkx.dynamics.Euler
  nkx.dynamics.Heun
  nkx.dynamics.Midpoint
  nkx.dynamics.RK12
  nkx.dynamics.RK23
  nkx.dynamics.RK4
  nkx.dynamics.RK45