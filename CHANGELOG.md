
```{currentmodule} netket
```

# Change Log

## NetKet 3.0b4 (unreleased)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b3...master).

### New features


### Breaking Changes


### Internal Changes


### Bug Fixes


## NetKet 3.0b3 (published on 9 july 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b2...v3.0b3).

### New features

* The {ref}`utils.group` submodule provides utilities for geometrical and permutation groups. `Lattice` (and its specialisations like `Grid`) use these to automatically construct the space groups of lattices, as well as their character tables for generating wave functions with broken symmetry. [#724](https://github.com/netket/netket/pull/724)
* Autoregressive neural networks, sampler, and masked linear layers have been added to `models`, `sampler` and `nn` [#705](https://github.com/netket/netket/pull/705).


### Breaking Changes

* The `graph.Grid` class has been removed. {ref}`graph.Grid` will now return an instance of {ref}`graph.Lattice` supporting the same API but with new functionalities related to spatial symmetries. The `color_edges` optional keyword argument has been removed without deprecation. [#724](https://github.com/netket/netket/pull/724)
* `MCState.n_discard` has been renamed `MCState.n_discard_per_chain` and the old binding has been deprecated [#739](https://github.com/netket/netket/pull/739).
* `nk.optimizer.qgt.QGTOnTheFly` option `centered=True` has been removed because we are now convinced the two options yielded equivalent results. `QGTOnTheFly` now always behaves as if `centered=False` [#706](https://github.com/netket/netket/pull/706).

### Internal Changes

* `networkX` has been replaced by `igraph`, yielding a considerable speedup for some graph-related operations [#729](https://github.com/netket/netket/pull/729).
* `netket.hilbert.random` module now uses `plum-dispatch` (through `netket.utils.dispatch`) to select the correct implementation of `random_state` and `flip_state`. This makes it easy to define new hilbert states and extend their functionality easily.  [#734](https://github.com/netket/netket/pull/734).
* The AbstractHilbert interface is now much smaller in order to also support continuous Hilbert spaces. Any functionality specific to discrete hilbert spaces (what was previously supported) has been moved to a new abstract type `nk.hilbert.DiscreteHilbert`. Any Hilbert space previously subclassing {ref}`nk.hilbert.AbstractHilbert` should be modified to subclass {ref}`nk.hilbert.DiscreteHilbert` [#800](https://github.com/netket/netket/pull/800).

### Bug Fixes

* `nn.to_array` and `MCState.to_array`, if `normalize=False`, do not subtract the logarithm of the maximum value from the state  [#705](https://github.com/netket/netket/pull/705).
* Autoregressive networks now work with Fock space and give correct errors if the hilbert space is not supported  [#806](https://github.com/netket/netket/pull/806).
* Autoregressive networks are now much (x10-x100) faster  [#705](https://github.com/netket/netket/pull/705).
* Do not throw errors when calling `operator.get_conn_flattened(states)` with a jax array  [#764](https://github.com/netket/netket/pull/764).
* Fix bug with the driver progress bar when `step_size != 1`  [#747](https://github.com/netket/netket/pull/747).


## NetKet 3.0b2 (published on 31 May 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b1...v3.0b2).

### New features

* Group Equivariant Neural Networks have been added to `models` [#620](https://github.com/netket/netket/pull/620)
* Permutation invariant RBM and Permutation invariant dense layer have been added to `models`
  and `nn.linear` [#573](https://github.com/netket/netket/pull/573)
* Add the property `acceptance` to `MetropolisSampler`'s `SamplerState`, computing the
  MPI-enabled acceptance ratio. [#592](https://github.com/netket/netket/pull/592).
* Add `StateLog`, a new logger that stores the parameters of the model during the
  optimization in a folder or in a tar file. [#645](https://github.com/netket/netket/pull/645)
* A warning is now issued if NetKet detects to be running under `mpirun` but MPI dependencies
  are not installed [#631](https://github.com/netket/netket/pull/631)
* `operator.LocalOperator`s now do not return a zero matrix element on the diagonal if the whole
  diagonal is zero. [#623](https://github.com/netket/netket/pull/623).
* `logger.JSONLog` now automatically flushes at every iteration if it does not consume significant
  CPU cycles. [#599](https://github.com/netket/netket/pull/599)
* The interface of Stochastic Reconfiguration has been overhauled and made more modular. You can now
  specify the solver you wish to use, NetKet provides some dense solvers out of the box, and there are
  3 different ways to compute the Quantum Geometric Tensor. Read the documentation to learn more about
  it. [#674](https://github.com/netket/netket/pull/674)
* Unless you specify the QGT implementation you wish to use with SR, we use an automatic heuristic based
  on your model and the solver to pick one.
  This might affect SR performance. [#674](https://github.com/netket/netket/pull/674)


### Breaking Changes

* For all samplers, `n_chains` now sets the _total_ number of chains across all MPI ranks. This is a breaking change
  compared to the old API, where `n_chains` would set the number of chains on a single MPI rank. It is still possible to
  set the number of chains per MPI rank by specifying `n_chains_per_rank` instead of `n_chains`. This change, while breaking
  allows us to be consistent with the interface of {ref}`variational.MCState`, where `n_samples` is the total number of samples
  across MPI nodes.
* `MetropolisSampler.reset_chain` has been renamed to `MetropolisSampler.reset_chains`.
  Likewise in the constructor of all samplers.
* Briefly during development releases `MetropolisSamplerState.acceptance_ratio` returned
  the percentage (not ratio) of acceptance. `acceptance_ratio` is now deprecated in
  favour of the correct `acceptance`.
* `models.Jastrow` now internally symmetrizes the matrix before computing its value [#644](https://github.com/netket/netket/pull/644)
* `MCState.evaluate` has been renamed to `MCState.log_value` [#632](https://github.com/netket/netket/pull/632)
* `nk.optimizer.SR` no longer accepts keyword argument relative to the sparse solver. Those should be passed
  inside the closure or `functools.partial` passed as `solver` argument.
* `nk.optimizer.sr.SRLazyCG` and `nk.optimizer.sr.SRLazyGMRES` have been deprecated and will soon be removed.
* Parts of the `Lattice` API have been overhauled, with deprecations of several methods in favor of a consistent usage of `Lattice.position` for real-space location of sites and `Lattice.basis_coords` for location of sites in terms of basis vectors. `Lattice.sites` has been added, which provides a sequence of `LatticeSite` objects combining all site properties. Furthermore, `Lattice` now provides lookup of sites from their position via `id_from_position` using a hashing scheme that works across periodic boundaries. [#703](https://github.com/netket/netket/pull/703) [#715](https://github.com/netket/netket/pull/715)
* `nk.variational` has been renamed to `nk.vqs` and will be removed in a future release.

### Bug Fixes

* Fix `operator.BoseHubbard` usage under jax Hamiltonian Sampling [#662](https://github.com/netket/netket/pull/662)
* Fix `SROnTheFly` for `R->C` models with non homogeneous parameters [#661](https://github.com/netket/netket/pull/661)
* Fix MPI Compilation deadlock when computing expectation values [#655](https://github.com/netket/netket/pull/655)
* Fix bug preventing the creation of a `hilbert.Spin` Hilbert space with odd sites and even `S`. [#641](https://github.com/netket/netket/pull/641)
* Fix bug [#635](https://github.com/netket/netket/pull/635) preventing the usage of `NumpyMetropolisSampler` with `MCState.expect` [#635](https://github.com/netket/netket/pull/635)
* Fix bug [#635](https://github.com/netket/netket/pull/635) where the `graph.Lattice` was not correctly computing neighbours because of floating point issues. [#633](https://github.com/netket/netket/pull/633)
* Fix bug the Y Pauli matrix, which was stored as its conjugate. [#618](https://github.com/netket/netket/pull/618) [#617](https://github.com/netket/netket/pull/617) [#615](https://github.com/netket/netket/pull/615)


## NetKet 3.0b1 (published beta release)

[GitHub commits](https://github.com/netket/netket/compare/v2.1.1...v3.0b1).

### API Changes

* Hilbert space constructors do not store the lattice graph anymore. As a consequence, the constructor does not accept the graph anymore.

* Special Hamiltonians defined on a lattice, such as {class}`operator.BoseHubbard`, {class}`operator.Ising` and {class}`operator.Heisenberg`, now require the graph to be passed explicitly through a `graph` keyword argument.

* {class}`operator.LocalOperator` now default to real-valued matrix elements, except if you construct them with a complex-valued matrix. This is also valid for operators such as :func:`operator.spin.sigmax` and similars.

* When performing algebraic operations {code}`*, -, +` on pairs of {class}`operator.LocalOperator`, the dtype of the result iscomputed using standard numpy promotion logic.

  * Doing an operation in-place {code}`+=, -=, *=` on a real-valued operator will now fail if the other is complex. While this might seem annoying, it's useful to ensure that smaller types such as `float32` or `complex64` are preserved if the user desires to do so.

* {class}`AbstractMachine` has been removed. It's functionality is now split among the model itself, which is defined by the user and {class}`variational.MCState` for pure states or {class}`variational.MCMixedState` for mixed states.

  * The model, in general is composed by two functions, or an object with two functions: an `init(rng, sample_val)` function, accepting a {ref}`jax.random.PRNGKey` object and an input, returning the parameters and the state of the model for that particular sample shape, and a {code}`apply(params, samples, **kwargs)` function, evaluating the model for the given parameters and inputs.

  * Some models (previously machines) such as the RBM (Restricted Boltzmann Machine) Machine, NDM (Neural Density Matrix) or MPS (Matrix Product State ansatz) are available in {ref}`Pre-built models`.

  * Machines, now called models, should be written using [Flax](https://flax.readthedocs.io/en/latest) or another jax framework.

  * Serialization and deserialization functionality has now been moved to {ref}`variational.MCState`, which support the standard Flax interface through MsgPack. See [Flax docs](https://flax.readthedocs.io/en/latest/flax.serialization.html) for more information

  * {code}`AbstractMachine.init_random_parameters` functionality has now been absorbed into {meth}`netket.vqs.VariationalState.init_parameters`, which however has a different syntax.

* {ref}`Samplers <Sampler>` now require the Hilbert space upon which they sample to be passed in to the constructor.
Also note that several keyword arguments of the samplers have changed, and new one are available.

* It's now possible to change {ref}`Samplers <Sampler>` dtype, which controls the type of the output. By default they use double-precision samples (`np.float64`). Be wary of type promotion issues with your models.

* {ref}`Samplers <Sampler>` no longer take a machine as an argument.

* {ref}`Samplers <Sampler>` are now immutable (frozen) `dataclasses` (defined through `flax.struct.dataclass`) that only hold the sampling parameters. As a consequence it is no longer possible to change their settings such as `n_chains` or `n_sweeps` without creating a new sampler. If you wish to update only one parameter, it is possible to construct the new sampler with the updated value by using the `sampler.replace(parameter=new_value)` function.

* {ref}`Samplers <Sampler>` are no longer stateful objects. Instead, they can construct an immutable state object {ref}`sampler.init_state`, which can be passed to sampling functions such as {ref}`sampler.sample`, which now return also the updated state. However, unless you have particular use-cases we advise you use the variational state {ref}`MCState` instead.

* The {ref}`Optimizer` module has been overhauled, and now only re-exports flax optim module. We advise not to use netket's optimizer but instead to use [optax](https://github.com/deepmind/optax>) .

* The {ref}`SR` object now is only a set of options used to compute the SR matrix. The SR matrix, now called `quantum_geometric_tensor` can be obtained by calling {meth}`variational.MCState.quantum_geometric_tensor`. Depending on the settings, this can be a lazy object.

* {ref}`netket.Vmc` has been renamed to {ref}`netkt.VMC`

* {ref}`netket.models.RBM` replaces the old {code}`RBM` machine, but has real parameters by default.

* As we rely on Jax, using {code}`dtype=float` or {code}`dtype=complex`, which are weak types, will sometimes lead to loss of precision because they might be converted to `float32`. Use {code}`np.float64` or {code}`np.complex128` instead if you want double precision when defining your models.
