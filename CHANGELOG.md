
```{currentmodule} netket
```

# Change Log

## NetKet 3.4 (⚙️ In development)

[GitHub commits](https://github.com/netket/netket/compare/v3.3...master).

### New features


### Breaking Changes


### Bug Fixes



## NetKet 3.3 (🎁 20 December 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.2...v3.3).

### New features
* The interface to define expectation and gradient function of arbitrary custom operators is now stable. If you want to define it for a standard operator that can be written as an average of local expectation terms, you can now define a dispatch rule for {ref}`netket.vqs.get_local_kernel_arguments` and {ref}`netket.vqs.get_local_kernel`. The old mechanism is still supported, but we encourage to use the new mechanism as it is more terse. [#954](https://github.com/netket/netket/pull/954)
* `nk.optimizer.Adam` now supports complex parameters, and you can use `nk.optimizer.split_complex` to make optimizers process complex parameters as if they are pairs of real parameters. [#1009](https://github.com/netket/netket/pull/1009)
* Chunking of `MCState.expect` and `MCState.expect_and_grad` computations is now supported, which allows to bound the memory cost in exchange of a minor increase in computation time. [#1006](https://github.com/netket/netket/pull/1006) (and discussions in [#918](https://github.com/netket/netket/pull/918) and [#830](https://github.com/netket/netket/pull/830))
* A new variational state that performs exact summation over the whole Hilbert space has been added. It can be constructed with {ref}`nk.vqs.ExactState` and supports the same Jax neural networks as {ref}`nk.vqs.MCState`. [#953](https://github.com/netket/netket/pull/953)
* `DenseSymm` allows multiple input features. [#1030](https://github.com/netket/netket/pull/1030)
* [Experimental] A new time-evolution driver  {ref}`nk.experimental.TDVP` using the time-dependent variational principle (TDVP) has been added. It works with time-independent and time-dependent Hamiltonians and Liouvillians. [#1012](https://github.com/netket/netket/pull/1012)
* [Experimental] A set of JAX-compatible Runge-Kutta ODE integrators has been added for use together with the new TDVP driver. [#1012](https://github.com/netket/netket/pull/1012)

### Breaking Changes
* The method `sample_next` in `Sampler` and exact samplers (`ExactSampler` and `ARDirectSampler`) is removed, and it is only defined in `MetropolisSampler`. The module function `nk.sampler.sample_next` also only works with `MetropolisSampler`. For exact samplers, please use the method `sample` instead. [#1016](https://github.com/netket/netket/pull/1016)
* The default value of `n_chains_per_rank` in `Sampler` and exact samplers is changed to 1, and specifying `n_chains` or `n_chains_per_rank` when constructing them is deprecated. Please change `chain_length` when calling `sample`. For `MetropolisSampler`, the default value is changed from `n_chains = 16` (across all ranks) to `n_chains_per_rank = 16`. [#1017](https://github.com/netket/netket/pull/1017)
* `GCNN_Parity` allowed biasing both the parity-preserving and the parity-flip equivariant layers. These enter into the network output the same way, so having both is redundant and makes QGTs unstable. The biases of the parity-flip layers are now removed. The previous behaviour can be restored using the deprecated `extra_bias` switch; we only recommend this for loading previously saved parameters. Such parameters can be transformed to work with the new default using `nk.models.update_GCNN_parity`. [#1030](https://github.com/netket/netket/pull/1030)
* Kernels of `DenseSymm` are now three-dimensional, not two-dimensional. Parameters saved from earlier implementations can be transformed to the new convention using `nk.nn.update_dense_symm`. [#1030](https://github.com/netket/netket/pull/1030)

### Deprecations
* The method `Sampler.samples` is added to return a generator of samples. The module functions `nk.sampler.sampler_state`, `reset`, `sample`, `samples`, and `sample_next` are deprecated in favor of the corresponding class methods. [#1025](https://github.com/netket/netket/pull/1025)
* Kwarg `in_features` of `DenseEquivariant` is deprecated; the number of input features are inferred from the input. [#1030](https://github.com/netket/netket/pull/1030)
* Kwarg `out_features` of `DenseEquivariant` is deprecated in favour of `features`. [#1030](https://github.com/netket/netket/pull/1030)

### Internal Changes
* The definitions of `MCState` and `MCMixedState` have been moved to an internal module, `nk.vqs.mc` that is hidden by default. [#954](https://github.com/netket/netket/pull/954)
* Custom deepcopy for `LocalOperator` to avoid building `LocalOperator` from scratch each time it is copied [#964](https://github.com/netket/pull/964)

### Bug Fixes
* The constructor of `TensorHilbert` (which is used by the product operator `*` for inhomogeneous spaces) no longer fails when one of the component spaces is non-indexable. [#1004](https://github.com/netket/netket/pull/1004)
* The {ref}`nk.hilbert.random.flip_state` method used by `MetropolisLocal` now throws an error when called on a {ref}`nk.hilbert.ContinuousHilbert` hilbert space instead of entering an endless loop. [#1014](https://github.com/netket/netket/pull/1014)
* Fixed bug in conversion to qutip for `MCMixedState`, where the resulting shape (hilbert space size) was wrong. [#1020](https://github.com/netket/netket/pull/1020)
* Setting `MCState.sampler` now recomputes `MCState.chain_length` according to `MCState.n_samples` and the new `sampler.n_chains`. [#1028](https://github.com/netket/netket/pull/1028)
* `GCNN_Parity` allowed biasing both the parity-preserving and the parity-flip equivariant layers. These enter into the network output the same way, so having both is redundant and makes QGTs unstable. The biases of the parity-flip layers are now removed. [#1030](https://github.com/netket/netket/pull/1030)


## NetKet 3.2 (26 November 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.1...v3.2).

### New features
* `GraphOperator` (and `Heisenberg`) now support passing a custom mapping of graph nodes to Hilbert space sites via the new `acting_on_subspace` argument. This makes it possible to create `GraphOperator`s that act on a subset of sites, which is useful in composite Hilbert spaces. [#924](https://github.com/netket/netket/pull/924)
* `PauliString` now supports any Hilbert space with local size 2. The Hilbert space is now the optional first argument of the constructor. [#960](https://github.com/netket/netket/pull/960)
* `PauliString` now can be multiplied and summed together, performing some simple algebraic simplifications on the strings they contain. They also lazily initialize their internal data structures, making them faster to construct but slightly slower the first time that their matrix elements are accessed. [#955](https://github.com/netket/netket/pull/955)
* `PauliString`s can now be constructed starting from an `OpenFermion` operator. [#956](https://github.com/netket/netket/pull/956)
* In addition to nearest-neighbor edges, `Lattice` can now generate edges between next-nearest and, more generally, k-nearest neighbors via the constructor argument `max_neighbor_order`. The edges can be distinguished by their `color` property (which is used, e.g., by `GraphOperator` to apply different bond operators). [#970](https://github.com/netket/netket/pull/970)
* Two continuous-space operators (`KineticEnergy` and `PotentialEnergy`) have been implemented. [#971](https://github.com/netket/netket/pull/971)
* `Heisenberg` Hamiltonians support different coupling strengths on `Graph` edges with different colors. [#972](https://github.com/netket/netket/pull/972).
* The `little_group` and `space_group_irreps` methods of `SpaceGroupBuilder` take the wave vector as either varargs or iterables. [#975](https://github.com/netket/netket/pull/975)
* A new `netket.experimental` submodule has been created and all experimental features have been moved there. Note that in contrast to the other `netket` submodules, `netket.experimental` is not imported by default. [#976](https://github.com/netket/netket/pull/976)

### Breaking Changes
* Moved `nk.vqs.variables_from_***` to `nk.experimental.vqs` module. Also moved the experimental samplers to `nk.sampler.MetropolisPt` and `nk.sampler.MetropolisPmap` to `nk.experimental.sampler`. [#976](https://github.com/netket/netket/pull/976)
* `operator.size`, has been deprecated. If you were using this function, please transition to `operator.hilbert.size`. [#985](https://github.com/netket/netket/pull/985)

### Bug Fixes
* A bug where `LocalOperator.get_conn_flattened` would read out-of-bounds memory has been fixed. It is unlikely that the bug was causing problems, but it triggered warnings when running Numba with boundscheck activated. [#966](https://github.com/netket/netket/pull/966)
* The dependency `python-igraph` has been updated to `igraph` following the rename of the upstream project in order to work on conda. [#986](https://github.com/netket/netket/pull/986)
* {attr}`~netket.vqs.MCState.n_samples_per_rank` was returning wrong values and has now been fixed. [#987](https://github.com/netket/netket/pull/987)
* The `DenseSymm` layer now also accepts objects of type `HashableArray` as `symmetries` argument. [#989](https://github.com/netket/netket/pull/989)
* A bug where `VMC.info()` was erroring has been fixed. [#984](https://github.com/netket/netket/pull/984)

## NetKet 3.1 (20 October 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0...v3.1).

### New features
* Added Conversion methods `to_qobj()` to operators and variational states, that produce QuTiP's qobjects.
* A function `nk.nn.activation.reim` has been added that transforms a nonlinearity to act seperately on the real and imaginary parts
* Nonlinearities `reim_selu` and `reim_relu` have been added
* Autoregressive Neural Networks (ARNN) now have a `machine_pow` field (defaults to 2) used to change the exponent used for the normalization of the wavefunction. [#940](https://github.com/netket/netket/pull/940).

### Breaking Changes
* The default initializer for `netket.models.GCNN` has been changed to from `jax.nn.selu` to `netket.nn.reim_selu` [#892](https://github.com/netket/netket/pull/892)
* `netket.nn.initializers` has been deprecated in favor of `jax.nn.initializers` [#935](https://github.com/netket/netket/pull/935).
* Subclasses of `AbstractARNN` must define the field `machine_pow` [#940](https://github.com/netket/netket/pull/940)
* `nk.hilbert.HilbertIndex` and `nk.operator.spin.DType` are now unexported (they where never intended to be visible).  [#904](https://github.com/netket/netket/pull/904)
* `AbstractOperator`s have been renamed `DiscreteOperator`s. `AbstractOperator`s still exist, but have almost no functionality and they are intended as the base class for more arbitrary (eg. continuous space) operators. If you have defined a custom operator inheriting from `AbstractOperator` you should change it to derive from `DiscreteOperator`. [#929](https://github.com/netket/netket/pull/929)


### Internal Changes
* `PermutationGroup.product_table` now consumes less memory and is more performant. This is helpfull when working with large symmetry groups. [#884](https://github.com/netket/netket/pull/884) [#891](https://github.com/netket/netket/pull/891)
* Added size check to `DiscreteOperator.get_conn` and throw helpful error messages if those do not match. [#927](https://github.com/netket/netket/pull/927)
* The internal `numba4jax` module has been factored out into a standalone library, named (how original) [`numba4jax`](http://github.com/PhilipVinc/numba4jax). This library was never intended to be used by external users, but if for any reason you were using it, you should switch to the external library. [#934](https://github.com/netket/netket/pull/934)
* `netket.jax` now includes several _batching_ utilities like `batched_vmap` and `batched_vjp`. Those can be used to build memory efficient batched code, but are considered internal, experimental and might change without warning. [#925](https://github.com/netket/netket/pull/925).


### Bug Fixes
* Autoregressive networks now work with `Qubit` hilbert spaces. [#937](https://github.com/netket/netket/pull/937)



## NetKet 3.0 (23 august 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b4...v3.0).

### New features


### Breaking Changes
* The default initializer for `netket.nn.Dense` layers now matches the same default as `flax.linen`, and it is `lecun_normal` instead of `normal(0.01)` [#869](https://github.com/netket/netket/pull/869)
* The default initializer for `netket.nn.DenseSymm` layers is now chosen in order to give variance 1 to every output channel, therefore defaulting to `lecun_normal` [#870](https://github.com/netket/netket/pull/870)

### Internal Changes


### Bug Fixes


## NetKet 3.0b4 (17 august 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b3...v3.0b4).

### New features
* DenseSymm now accepts a mode argument to specify whever the symmetries should be computed with a full dense matrix or FFT. The latter method is much faster for sufficiently large systems. Other kwargs have been added to satisfy the interface. The api changes are also reflected in RBMSymm and GCNN. [#792](https://github.com/netket/netket/pull/792)

### Breaking Changes
* The so-called legacy netket in `netket.legacy` has been removed. [#773](https://github.com/netket/netket/pull/773)

### Internal Changes
* The methods `expect` and `expect_and_grad` of `MCState` now use dispatch to select the relevant implementation of the algorithm. They can therefore be expanded and overridden without editing NetKet's source code. [#804](https://github.com/netket/netket/pull/804)
* `netket.utils.mpi_available` has been moved to `netket.utils.mpi.available` to have a more consistent api interface (all mpi-related properties in the same submodule). [#827](https://github.com/netket/netket/pull/827)
* `netket.logging.TBLog` has been renamed to `netket.logging.TensorBoardLog` for better readability. A deprecation warning is now issued if the older name is used [#827](https://github.com/netket/netket/pull/827)
* When `MCState` initializes a model by calling `model.init`, the call is now jitted. This should speed it up for non-trivial models but might break non-jit invariant models.  [#832](https://github.com/netket/netket/pull/832)
* `operator.get_conn_padded` now supports arbitrarily-dimensioned bitstrings as input and reshapes the output accordingly.  [#834](https://github.com/netket/netket/pull/834)
* NetKet's implementation of dataclasses now support `pytree_node=True/False` on cached properties. [#835](https://github.com/netket/netket/pull/835)
* Plum version has been bumped to 1.5.1 to avoid broken versions (1.4, 1.5). [#856](https://github.com/netket/netket/pull/856).
* Numba version 0.54 is now allowed [#857](https://github.com/netket/netket/pull/857).


### Bug Fixes
* Fix Progress bar bug. [#810](https://github.com/netket/netket/pull/810)
* Make the repr/printing of history objects nicer in the REPL. [#819](https://github.com/netket/netket/pull/819)
* The field `MCState.model` is now read-only, to prevent user errors. [#822](https://github.com/netket/netket/pull/822)
* The order of the operators in `PauliString` does no longer influences the estimate of the number of non-zero connected elements. [#836](https://github.com/netket/netket/pull/836)

## NetKet 3.0b3 (published on 9 july 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.0b2...v3.0b3).

### New features

* The {ref}`netket.utils.group` submodule provides utilities for geometrical and permutation groups. `Lattice` (and its specialisations like `Grid`) use these to automatically construct the space groups of lattices, as well as their character tables for generating wave functions with broken symmetry. [#724](https://github.com/netket/netket/pull/724)
* Autoregressive neural networks, sampler, and masked linear layers have been added to `models`, `sampler` and `nn` [#705](https://github.com/netket/netket/pull/705).


### Breaking Changes

* The `netket.graph.Grid` class has been removed. {ref}`netket.graph.Grid` will now return an instance of {ref}`graph.Lattice` supporting the same API but with new functionalities related to spatial symmetries. The `color_edges` optional keyword argument has been removed without deprecation. [#724](https://github.com/netket/netket/pull/724)
* `MCState.n_discard` has been renamed `MCState.n_discard_per_chain` and the old binding has been deprecated [#739](https://github.com/netket/netket/pull/739).
* `nk.optimizer.qgt.QGTOnTheFly` option `centered=True` has been removed because we are now convinced the two options yielded equivalent results. `QGTOnTheFly` now always behaves as if `centered=False` [#706](https://github.com/netket/netket/pull/706).

### Internal Changes

* `networkX` has been replaced by `igraph`, yielding a considerable speedup for some graph-related operations [#729](https://github.com/netket/netket/pull/729).
* `netket.hilbert.random` module now uses `plum-dispatch` (through `netket.utils.dispatch`) to select the correct implementation of `random_state` and `flip_state`. This makes it easy to define new hilbert states and extend their functionality easily.  [#734](https://github.com/netket/netket/pull/734).
* The AbstractHilbert interface is now much smaller in order to also support continuous Hilbert spaces. Any functionality specific to discrete hilbert spaces (what was previously supported) has been moved to a new abstract type `netket.hilbert.DiscreteHilbert`. Any Hilbert space previously subclassing {ref}`netket.hilbert.AbstractHilbert` should be modified to subclass {ref}`netket.hilbert.DiscreteHilbert` [#800](https://github.com/netket/netket/pull/800).

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

  * The model, in general is composed by two functions, or an object with two functions: an `init(rng, sample_val)` function, accepting a {func}`jax.random.PRNGKey` object and an input, returning the parameters and the state of the model for that particular sample shape, and a {code}`apply(params, samples, **kwargs)` function, evaluating the model for the given parameters and inputs.

  * Some models (previously machines) such as the RBM (Restricted Boltzmann Machine) Machine, NDM (Neural Density Matrix) or MPS (Matrix Product State ansatz) are available in {ref}`Pre-built models`.

  * Machines, now called models, should be written using [Flax](https://flax.readthedocs.io/en/latest) or another jax framework.

  * Serialization and deserialization functionality has now been moved to {ref}`netket.variational.MCState`, which support the standard Flax interface through MsgPack. See [Flax docs](https://flax.readthedocs.io/en/latest/flax.serialization.html) for more information

  * {code}`AbstractMachine.init_random_parameters` functionality has now been absorbed into {meth}`netket.vqs.VariationalState.init_parameters`, which however has a different syntax.

* {ref}`Samplers <Sampler>` now require the Hilbert space upon which they sample to be passed in to the constructor.
Also note that several keyword arguments of the samplers have changed, and new one are available.

* It's now possible to change {ref}`Samplers <Sampler>` dtype, which controls the type of the output. By default they use double-precision samples (`np.float64`). Be wary of type promotion issues with your models.

* {ref}`Samplers <Sampler>` no longer take a machine as an argument.

* {ref}`Samplers <Sampler>` are now immutable (frozen) `dataclasses` (defined through `flax.struct.dataclass`) that only hold the sampling parameters. As a consequence it is no longer possible to change their settings such as `n_chains` or `n_sweeps` without creating a new sampler. If you wish to update only one parameter, it is possible to construct the new sampler with the updated value by using the `sampler.replace(parameter=new_value)` function.

* {ref}`Samplers <Sampler>` are no longer stateful objects. Instead, they can construct an immutable state object {ref}`netket.sampler.init_state`, which can be passed to sampling functions such as {ref}`netket.sampler.sample`, which now return also the updated state. However, unless you have particular use-cases we advise you use the variational state {ref}`MCState` instead.

* The {ref}`netket.optimizer` module has been overhauled, and now only re-exports flax optim module. We advise not to use netket's optimizer but instead to use [optax](https://github.com/deepmind/optax>) .

* The {ref}`netket.optimizer.SR` object now is only a set of options used to compute the SR matrix. The SR matrix, now called `quantum_geometric_tensor` can be obtained by calling {meth}`variational.MCState.quantum_geometric_tensor`. Depending on the settings, this can be a lazy object.

* `netket.Vmc` has been renamed to {ref}`netket.VMC`

* {ref}`netket.models.RBM` replaces the old {code}`RBM` machine, but has real parameters by default.

* As we rely on Jax, using {code}`dtype=float` or {code}`dtype=complex`, which are weak types, will sometimes lead to loss of precision because they might be converted to `float32`. Use {code}`np.float64` or {code}`np.complex128` instead if you want double precision when defining your models.
