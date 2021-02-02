
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

   netket.graph.AbstractGraph
   netket.graph.NetworkX
   netket.graph.Edgeless
   netket.graph.Hypercube
   netket.graph.Lattice
   netket.graph.Chain
   netket.graph.Grid

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
~~~~~~~~~~~~~~~~~~~~~

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

   netket.sampler.Sampler
   netket.sampler.ExactSampler
   netket.sampler.MetropolisSampler
   netket.sampler.MetropolisNumpySampler
   netket.sampler.MetropolisPtSampler

   netket.sampler.MetropolisLocal
   netket.sampler.MetropolisExchange
   netket.sampler.MetropolisHamiltonian
   netket.sampler.MetropolisLocalPt
   netket.sampler.MetropolisExchangePt

Transition Rules
~~~~~~~~~~~~~~~~

Those are the transition rules that can be used with the Metropolis
Sampler. 

.. autosummary::
  :toctree: _generated/samplers

  netket.sampler.MetropolisRule
  netket.sampler.rules.LocalRule
  netket.sampler.rules.ExchangeRule
  netket.sampler.rules.HamiltonianRule
  netket.sampler.rules.HamiltonianRuleNumpy
  netket.sampler.rules.CustomRuleNumpy


Internal State
~~~~~~~~~~~~~~

.. autosummary::
  :toctree: _generated/samplers

  netket.sampler.SamplerState
  netket.sampler.MetropolisSamplerState

Pre-built models
----------------

.. autosummary::
   :toctree: _generated/models
   :nosignatures:

   netket.models.RBM
   netket.models.RBMModPhase
   netket.models.MPSPeriodic


Model tools
----------------

This sub-module wraps and re-exports `flax.nn <https://flax.readthedocs.io/en/latest/flax.linen.html>`_.
Read more about the design goal of this module in their `README <https://github.com/google/flax/blob/master/flax/linen/README.md>`_

.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   netket.nn.Module

Linear Modules
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: _generated/nn
   :nosignatures:

   netket.nn.Dense
   netket.nn.DenseGeneral
   netket.nn.Conv
   netket.nn.Embed


Activation functions
~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: netket.nn

.. autosummary::
  :toctree: _generated/nn

    celu
    elu
    gelu
    glu
    log_sigmoid
    log_softmax
    relu
    sigmoid
    soft_sign
    softmax
    softplus
    swish
    logcosh


Variational State Interface
---------------------------

.. currentmodule:: netket

.. autosummary::
  :toctree: _generated/variational
  :nosignatures:

  netket.variational.VariationalState
  netket.variational.MCState
  netket.variational.MCMixedState



Optim
---------

.. currentmodule:: netket

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

