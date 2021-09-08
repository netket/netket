*******************************
The Variational State Interface
*******************************

A central element of NetKet is the :class:`~netket.vqs.VariationalState` interface.
A Variational State represents an approximate, variational description of a system, and can be used to
probe it.

As of now, NetKet has two types of Variational state implementations:

- :class:`~netket.vqs.MCState`, which is a classical variational approximation of a pure state.

- :class:`~netket.vqs.MCMixedState`, which is a classical variational approximation of a mixed state.


It is our plan to build upon this interface for new kind of variational states in the future. For example, we
are interested in implementing a new kind of variational state that encodes a state into a Qiskit circuit.

Constructing a Variational State
--------------------------------

To construct the two variational states above you need to provide at least a Monte Carlo sampler (you can see
the list of available ones :ref:`here <sampler-api>`) and a model.
The hilbert space of the variational state will be inferred from the sampler, as they all reference the
hilbert space they are sampling.

The model can be specified in several ways. The most standard way is to pass a `Flax Linen Module <https://flax.readthedocs.io/en/latest/flax.linen.html#module>`_, but you can also pass a Jax-style pair
of functions :code:`(init, apply)` or an haiku module obtained by calling :code:`haiku.transform()`.
It's also easy to use another jax-based framework with NetKet (though it's not yet documented). If you
want to do that, you can have a look at the automatic conversion `code <https://github.com/netket/netket/tree/master/netket/utils/model_frameworks>`_ and get in touch with us.

.. warning::

    Initializing a Variational State with non-Flax modules is still experimental. Complex modules,
    especially if mutable, might not work. If you have problems don't hesitate opening issues on
    GitHub so that we can address them. They are mostly easy fixes.


Defining models: Flax
^^^^^^^^^^^^^^^^^^^^^

In general, the most-well supported way to define Models for NetKet is through Flax.

If you are not familiar with Flax, it's a package that allows to define Neural Networks or other
parametrized functions. While we suggest to read carefully the `introduction to Flax <https://flax.readthedocs.io/en/latest/notebooks/flax_basics.html>`_, if you are impatient and just want to define some simple, easy
models you can find some examples in the `Model Surgery <https://flax.readthedocs.io/en/latest/howtos/model_surgery.html>`_ section of Flax documentation.

Other examples can be found in the source of the :ref:`Pre-built models <_Models>` distributed with NetKet, such as
:class:`~netket.models.RBM` (the simplest one), :class:`~netket.models.MPSPeriodic` and :class:`~netket.models.NDM` for
more complicated examples.

.. warning::

    Flax modules do not always work well with complex numbers. For that reason, if your model has complex
    parameters we invite you to use :code:`netket.nn` in place of :code:`flax.linen` and :code:`flax.nn`.

    :code:`netket.nn` re-exports some Flax functionality ensuring that it works nicely with complex numbers.
    If you don't find something from Flax in :code:`netket.nn`, please `open an issue <https://github.com/netket/netket/issues>`_ in GitHub.


MCState
^^^^^^^

Let's now assume we have a Flax module, :class:`~netket.models.RBM` and we want to construct a variational state
on sampled with a simple :func:`~netket.sampler.MetropolisLocal` sampler.

To do this, first you need to construct the model and the sampler, and then you can pass them, together with several
parameters, to the constructor.

.. code:: python

    hilbert = nk.hilbert.Spin(0.5)**10

    sampler = nk.sampler.MetropolisLocal(hilbert)

    model = nk.models.RBM(alpha=1, dtype=float, kernel_init=nk.nn.initializers.normal(stddev=0.01))

    vstate = nk.vqs.MCState(sampler, model, n_samples=500)

When constructed, the variational state will call the model's init method to generate the state and the
parameters, and will also initialize the sampler.

When constructing a variational state you can also pass a seed, that will be used to initialize both the
weights and the sampler. You can also pass two different seeds for the sampler and the weights.


Using a Monte Carlo Variational State
-------------------------------------

Expectation values
^^^^^^^^^^^^^^^^^^

One you have a variational state, you can do many things with it.
First of all, you can probe expectation values:

.. code:: python

    Ĥ = nk.operator.Ising(hilbert, nk.graph.Chain(hilbert.size), h=0.5)

    vstate.expect(Ĥ)

    >>> -4.98 ± 0.14 [σ²=9.51, R̂=1.0006]

Notice that if you call multiple times :code:`expect`, the same set of
samples will be used, and you will get the same result. To force sampling
to happen again, you can call :py:meth:`~netket.vqs.MCState.sample`.

.. code:: python

    vstate.expect(Ĥ)

    >>> -4.98 ± 0.14 [σ²=9.51, R̂=1.0006]

    vstate.sample();

    vstate.expect(Ĥ)

    >>> -4.90 ± 0.14 [σ²=9.54, R̂=1.0062]

The set of the last sampled samples can be accessed from the attribute
:py:attr:`~netket.vqs.MCState.samples`. If you access the samples
from this attribute, but you haven't sampled already, :code:`sample()` will
be called automatically.

Note that when you update the parameters, the samples are automatically
discarded.

Parameters can be accessed through the attribute :py:attr:`~netket.vqs.VariationalState.parameters`,
and you can modify them by assigning a new set of parameters to this attribute.

Note that parameters cannot in general be modified in place, as they are
of type `FrozenDict` (they are frozen, aka can't be modified). A typical way
to modify them, for example to add 0.1 to all parameters is to do the following:

.. code:: python

    import jax

    # See the value of some parameters
    vstate.parameters['visible_bias']

    >>> DeviceArray([-0.10806808, -0.14987472,  0.13069461,  0.0125838 ,
                  0.06278384,  0.00275547,  0.05843748,  0.07516951,
                  0.21897993, -0.01632223], dtype=float64)

    vstate.parameters = jax.tree_map(lambda x: x+0.1, vstate.parameters)

    # Look at the new values
    vstate.parameters['visible_bias']
    >>> DeviceArray([-0.00806808, -0.04987472,  0.23069461,  0.1125838 ,
                  0.16278384,  0.10275547,  0.15843748,  0.17516951,
                  0.31897993,  0.08367777], dtype=float64)


Sampling
^^^^^^^^

You can also change the number of samples to extract (note: this will
trigger recompilation of the sample function, so you should not do this
in a hot loop) by changing :py:attr:`~netket.vqs.MCState.n_samples`, and
the number of discarded samples at the beginning of every markov chain by
changing :py:attr:`~netket.vqs.MCState.n_discard_per_chain`.

By default, :py:attr:`~netket.vqs.MCState.n_discard_per_chain` is 10% of
:py:attr:`~netket.vqs.MCState.n_samples`.

The number of samples is then split among the number of chains/batches of the sampler.

.. code:: python

    hilbert = nk.hilbert.Spin(0.5)**6

    sampler = nk.sampler.MetropolisLocal(hilbert, n_chains=8)

    vstate = nk.vqs.MCState(sampler, nk.models.RBM(), n_samples=500)

    print(vstate.n_samples)
    504

    print(vstate.chain_length)
    63

    print(vstate.n_discard_per_chain)
    50

You can see that 500 samples are split among 8 chains, giving :math:`500/8=62.5` (rounded to
the next largest integer, 63). Therefore 8 chains of length 63 will be run.
n_discard_per_chain gives the number of discarded steps taken in the markov chain before actually storing
them, so the Markov Chains are actually :code:`chain_length + n_discard_per_chain` long. The default
n_discard_per_chain is 10% of the total samples, but you can change that to any number.

.. _warn-mpi-sampling:

.. warning::

    When running your code under MPI, the length of the chain is computed not only by dividing the
    total number of samples by :code:`n_chains`, but also by diving it by the number of MPI processes.

    Therefore, considering the number from the example above, if we had 4 MPI processes, we would have
    found a chain length of :math:`500/(8*4) = 15.625 \rightarrow 16`.

Collecting the state-vector
^^^^^^^^^^^^^^^^^^^^^^^^^^^

A variational state can be evaluated on the whole Hilbert space in order to obtain
the ket it represents.

This is achieved by using the :py:meth:`~netket.vqs.VariationalState.to_array` method,
which by defaults normalises the :math:`L_2` norm of the vector to 1 (but can be turned off).

Mixed state ansatzes can be converted to their matrix representation with
:py:meth:`~netket.vqs.MCMixedState.to_matrix`. In this case, the default
normalisation sets the trace to 1.


Manipulating the parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can access the parameters of a variational state through the :py:attr:`~netket.vqs.VariationalState.parameters` attribute.
Similarly, if your model has also a mutable state, you can access it through
the :py:attr:`~netket.vqs.VariationalState.model_state` attribute.

Parameters are stored as a Flax :code:`FrozenDict`, which behaves like a standard python dictionary but cannot be modified.
In Jax jargon, Parameters are a PyTree (see `PyTree documentation <https://jax.readthedocs.io/en/latest/pytrees.html>`_) and they
can be operated upon with functions like `jax.tree_map <https://jax.readthedocs.io/en/latest/jax.tree_util.html?highlight=tree_map#jax.tree_util.tree_map>`_.

You can also modify the parameters by _unfreezing_ them, using the command `flax.core.unfreeze`.

.. code:: python

    import flax

    pars = flax.core.unfreeze(varstate.parameters)

    pars['Dense']['kernel'] = pars['Dense']['kernel'] +1

    varstate.parameters = pars

The parameter dict will be automatically frozen upon assignment.


Saving and Loading a Variational State
--------------------------------------

Variational States conform to the `Flax serialization interface <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_ and can be serialized and deserialized with it.

Moreover, the Json Logger :class:`~netket.logging.JsonLog` serializes the variational state
through that interface.

A simple example to serialize data is provided below:

.. code:: python

    # construct an RBM model on 10 spins
    vstate = nk.vqs.MCState(nk.sampler.MetropolisLocal(nk.hilbert.Spin(0.5)**10),
                                    nk.models.RBM())

    import flax

    with open("test.mpack", 'wb') as file:
      file.write(flax.serialization.to_bytes(vstate))


And here we de-serialize it:

.. code:: python

    # construct a new RBM model on 10 spins
    vstate = nk.vqs.MCState(nk.sampler.MetropolisLocal(nk.hilbert.Spin(0.5)**10),
                                    nk.models.RBM())

    # load
    with open("test.mpack", 'rb') as file:
      vstate = flax.serialization.from_bytes(vstate, file.read())

Note that this also serializes the state of the sampler.

.. note::

    The JSonLog serializer only serializes the parameters of the model, and not the whole variational state.
    Therefore, if you wish to reload the parameters of a variational state, saved by the json logger, you should
    use the same procedure outlined above, only that the list line should be:

    .. code:: python

        with open("parameters.mpack", 'rb') as file:
          vstate.variables = flax.serialization.from_bytes(vstate.variables, file.read())
