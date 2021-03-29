########################
What's New in v3.0
########################

.. currentmodule:: netket

In this page we discuss the main differences between old versions of netket and version 3.0.

API Changes
###########


* Hilbert space constructors do not store the lattice graph anymore. As a consequence, the constructor does not accept the graph anymore.

* Special Hamiltonians defined on a lattice, such as :class:`operator.BoseHubbard`, :class:`operator.Ising` and :class:`operator.Heisenberg`, now require the graph to be passed explicitly through a `graph` keyword argument.

* :class:`operator.LocalOperator` now default to real-valued matrix elements, except if you construct them with a complex-valued matrix. This is also valid for operators such as :func:`operator.spin.sigmax` and similars.

* When performing algebric operations :code:`*, -, +` on pairs of :class:`operator.LocalOperator`, the dtype of the result iscomputed using standard numpy promotion logic. 

  * Doing an operation in-place :code:`+=, -=, *=` on a real-valued operator will now fail if the other is complex. While this might seem annoying, it's usefull to ensure that smaller types such as `float32` or `complex64` are preserved if the user desires to do so.

* :class:`AbstractMachine` has been removed. It's functionality is now split among the model itself, which is defined by the user and :class:`variational.MCState` for pure states or :class:`variational.MCMixedState` for mixed states.
	
  * The model, in general is composed by two functions, or an object with two functions: an `init(rng, sample_val)` function, accepting a :ref:`jax.random.PRNGKey` object and an input, returning the parameters and the state of the model for that particular sample shape, and a :code:`apply(params, samples, **kwargs)` function, evaluating the model for the given parameters and inputs.

  * Some models (previously machines) such as the RBM (Restricted Bolzmann Machine) Machine, NDM (Neural Density Matrix) or MPS (Matrix Product State ansatz) are available in :ref:`Pre-built models`. 

  * Machines, now called models, should be written using `flax <https://flax.readthedocs.io/en/latest>`_ or another jax framework.

  * Serialization and deserialization functionality has now been moved to :ref:`variational.MCState`, which support the standard Flax interface through MsgPack. See `Flax docs <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_ for more information  

  * :code:`AbstractMachine.init_random_parameters` functionality has now been absorbed into :py:meth:`netket.variational.VariationalState.init_parameters`, which however has a different syntax.

* :ref:`Samplers <Sampler>` now require the hilbert space upon which they sample to be passed in to the constructor.
Also note that several keyword arguments of the samplers have changed, and new one are available.

* It's now possible to change :ref:`Samplers <Sampler>` dtype, which controls the type of the output. By default they use double-precision samples (`np.float64`). Be wary of type promotion issues with your models.
	
* :ref:`Samplers <Sampler>` no longer take a machine as an argument.

* :ref:`Samplers <Sampler>` are now immutable (frozen) `dataclasses` (defined through `flax.struct.dataclass`) that only hold the sampling parameters. As a consequence it is no longer possible to change their settings such as `n_chains` or `n_sweeps` without creating a new sampler. If you wish to update only one parameter, it is possible to construct the new sampler with the updated value by using the `sampler.replace(parameter=new_value)` function. 

* :ref:`Samplers <Sampler>` are no longer stateful objects. Instead, they can construct an immutable state object :ref:`sampler.init_state`, which can be passed to sampling functions such as :ref:`sampler.sample`, which now return also the updated state. However, unless you have particoular use-cases we advise you use the variational state :ref:`MCState` instead.

* The :ref:`Optimizer` module has been overhauled, and now only re-exports flax optim module. We advise not to use netket's optimizer but instead to use `optax <https://github.com/deepmind/optax>`_ .

* The :ref:`SR` object now is only a set of options used to compute the SR matrix. The SR matrix, now called `quantum_geometric_tensor` can be optained by calling :ref:`MCState.quantum_geometric_tensor(sr)`. Depending on the settings, this can be a lazy object.

* :ref:`netket.Vmc` has been renamed to :ref:`netkt.VMC`

* :ref:`netket.models.RBM` replaces the old :code:`RBM` machine, but has real parameters by default.

* As we rely on Jax, using :code:`dtype=float` or :code:`dtype=complex`, which are weak types, will sometimes lead to loss of precision because they might be converted to `float32`. Use :code:`np.float64` or :code:`np.complex128` instead if you want double precision when defining your models.
