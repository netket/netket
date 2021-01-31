
.. _api:

##########################
Public API: netket package
##########################

.. currentmodule:: netket


.. _graph-api:

Graph
-----

.. autosummary::
   :toctree: _generated/graph
   :nosignatures:

   netket.graph.Edgeless
   netket.graph.Hypercube
   netket.graph.Lattice

.. _hilbert-api:

Hilbert
-------

.. autosummary::
   :toctree: _generated/hilbert
   :nosignatures:

   netket.hilbert.AbstractHilbert
   netket.hilbert.Qubit
   netket.hilbert.Spin
   netket.hilbert.Boson
   netket.hilbert.CustomHilbert
   netket.hilbert.DoubledHilbert

.. _operators-api:

Operators
---------

.. autosummary::
   :toctree: _generated/operator
   :nosignatures:

   netket.operator.BoseHubbard
   netket.operator.GraphOperator
   netket.operator.LocalOperator
   netket.operator.Ising
   netket.operator.Heisenberg


Pre-defined operators
---------------------

.. autosummary::
   :toctree: _generated/hilbert
   :nosignatures:

   netket.operator.boson.create
   netket.operator.boson.destroy
   netket.operator.boson.number
   netket.operator.spin.sigmax
   netket.operator.spin.sigmay
   netket.operator.spin.sigmaz
   netket.operator.spin.sigmap
   netket.operator.spin.sigmam

Exact solvers
-------------

.. autosummary::
   :toctree: _generated/exact
   :nosignatures:

   netket.exact.full_ed
   netket.exact.lanczos_ed
   netket.exact.steady_state


Samplers
--------

.. autosummary::
   :toctree: _generated/samplers
   :nosignatures:

   netket.sampler.ExactSampler
   netket.sampler.Metropolis
   netket.sampler.MetropolisNumpy
   netket.sampler.MetropolisPt

   netket.sampler.MetropolisLocal
   netket.sampler.MetropolisExchange
   netket.sampler.MetropolisHamiltonian
   netket.sampler.MetropolisLocalPt
   netket.sampler.MetropolisExchangePt



Optim
---------


This module provides the following functionalities
.. autosummary::
   :toctree: _generated/optim
   :nosignatures:
   
   netket.optim.SR

This module also re-exports `flax.optim <https://flax.readthedocs.io/en/latest/flax.optim.html#available-optimizers>`_. Check it out for up-to-date informations on available optimisers. 
A non-comprehensive list is also included here:

.. autosummary::
   :toctree: _generated/optim
   :nosignatures:

   netket.optim.Adam
   netket.optim.Adagrad
   netket.optim.GradientDescent
   netket.optim.LAMB
   netket.optim.LARS
   netket.optim.Momentum
   netket.optim.RMSProp

