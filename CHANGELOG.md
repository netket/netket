```{currentmodule} netket
```

# Change Log

## NetKet 3.12 (⚙️ In development)

### New Features
* Discrete Hilbert spaces now use a special {class}`nk.utils.StaticRange` object to store the local values that label the local degree of freedom. This special object is jax friendly and can be converted to arrays, and allows for easy conversion from the local degrees of freedom to integers that can be used to index into arrays, and back. While those objects are not really used internally yet, in the future they will be used to simplify the implementations of operators and other objects [#1732](https://github.com/netket/netket/issues/1732).
* Some utilities to time execution of training loop are now provided, that can be used to coarsely see what part of the algorithm is dominating the training cost. To use it, pass `driver.run(..., timeit=True)` to all drivers when running them.
* Added several new tensor network ansatze to the `nk.models.tensor_networks` namespace. Those also replace previous tensor network implementations, that were de-facto broken [#1745](https://github.com/netket/netket/issues/1745).
* Add jax implementation of Bose Hubbard Operator, named {class}`netket.operator.BoseHubbardJax` and split numba implementation in a separate class [#1773](https://github.com/netket/netket/issues/1773).
* NetKet now automatically sets the visible GPUs when running under MPI with GPUs, by enumerating local GPUs and setting `jax_default_device` according to some local rank. This behaviour should allow users to not have to specify `CUDA_VISIBLE_DEVICES` and local mpi ranks on their scripts. This behaviour is only activated when running using MPI, and not used when using experimental sharding mode. To disable this functionality, set `NETKET_MPI_AUTODETECT_LOCAL_GPU=0` [#1757](https://github.com/netket/netket/issues/1757).
* {class}`netket.experimental.models.Slater2nd` now implements also the generalized hartree fock, as well as the restricted and unrestricted HF of before [#1765](https://github.com/netket/netket/issues/1765).
* A new variational state computing the sum of multiple slater determinants has been added, named {class}`nk.experimental.models.MultiSlater2nd`. This state has the same options of {class}`~netket.experimental.models.Slater2nd` [#1765](https://github.com/netket/netket/issues/1765).


### Breaking Changes
* The `out` keyword of Discrete Hilbert indexing methods (`all_states`, `numbers_to_states` and `states_to_numbers`) deprecated in the last release has been removed completely [#1722](https://github.com/netket/netket/issues/1722).
* The Homogeneous Hilbert spaces now must store the list of valid local values for the states with a {class}`nk.utils.StaticRange` objects instead of list of floats. The constructors have been updated accordingly. {class}`~nk.utils.StaticRange` is a range-like object that is jax-compatible and from now on should be used to index into local hilbert spaces [#1732](https://github.com/netket/netket/issues/1732).
* The `numbers_to_states` and `states_to_numbers` methods of {class}`netket.hilbert.DiscreteHilbert` must now be jax jittable. Custom Hilbert spaces using non-jittable functions have to be adapted by including a {func}`jax.pure_callback` in the `numbers_to_states`/`states_to_numbers` member functions [#1748](https://github.com/netket/netket/issues/1748).

### Deprecations
* The method {func}`netket.nn.states_to_numbers` is now deprecated. Please use {meth}`~DiscreteHilbert.numbers_to_states` directly.

### Improvements
* Rewrite the code for generating random states of `netket.hilbert.Fock` and `netket.hilbert.Spin` in Jax and jit the `init` and `reset` functions of `netket.sampler.MetropolisSampler` for better performance and improved compatibility with sharding [#1721](https://github.com/netket/netket/pull/1721).
* Rewrite `netket.hilbert.index` used by `HomogeneousHilbert` (including `Spin` and `Fock`) so that larger spaces with a sum constraint can be indexed. This can be useful for `netket.sampler.Exactsampler`, `netket.vqs.FullSumState` as well as for ED calculations [#1720](https://github.com/netket/netket/pull/1720).
* Duplicating a `netket.vqs.MCState` now leads to perfectly deterministic, identical samples between two different copies of the same `MCState` even if the sampler is changed. Previously, duplicating an `MCState` and changing the sampler on two copies of the same state would lead to some completely random seed being used and therefore different samples to be generated. This change is needed to eventually achieve proper checkpointing of our calculations [#1778](https://github.com/netket/netket/pull/1778).
* The methods converting Jax Operators to another kind (such as LocalOperators to PauliOperators) will return the Jax version of those operators if available [#1781](https://github.com/netket/netket/pull/1781).

### Finalized Deprecations
* Removed module function `netket.sampler.sample_next` that was deprecated in NetKet 3.3 (December 2021) [#17XX](https://github.com/netket/netket/pull/17XX).

### Internal changes
* Initialize the MetropolisSamplerState in a way that avoids recompilation when using sharding [#1776](https://github.com/netket/netket/pull/1776).
* Wrap several functions in the samplers and operators with a `shard_map` to avoid unnecessary collective communication when doing batched indexing of sharded arrays [#1777](https://github.com/netket/netket/pull/1777).
* Callbacks are now Pytree and can be flattened/unflatted and serialized with flax [#1666](https://github.com/netket/netket/pull/1666).

### Bug Fixes
* Fixed the gradient of variational states w.r.t. complex parameters which was missing a factor of 2. The learning rate needs to be halved to reproduce simulations made with previous versions of NetKet [#1785](https://github.com/netket/netket/pull/1785).
* Fixed the bug [#1791](https://github.com/netket/netket/pull/1791). where MetropolisHamiltonian with jax operators was leaking tracers and crashing [#1792](https://github.com/netket/netket/pull/1792).


## NetKet 3.11.3 (🐟 2 April 2024)

Bugfix release addressing the following issues:
* Fixes a bug where the conjugate of a fermionic operator was the conjugate-transpose, and the hermitian transpose `.H` was the identity. This could break code relying on complex-valued fermionic operators [#1743](https://github.com/netket/netket/pull/1743).
* Fixed a bug when converting jax operators to qutip format [#1749](https://github.com/netket/netket/pull/1749).
* Fixed an internal bug of `netket.utils.struct.Pytree`, where the cached properties's cache was not cleared when `replace` was used to copy and modify the Pytree [#1750](https://github.com/netket/netket/pull/1750).
* Update upper bound on optax to `optax<0.3`, following the release of `optax` 0.2 [#1751](https://github.com/netket/netket/pull/1751).
* Support QuTiP 5, released in march 2024 [#1762](https://github.com/netket/netket/pull/1762).


## NetKet 3.11.2 (27 february 2024)

Bugfix release to solve the following issues:
* Fix error thrown in repr method of error thrown in TDVP integrators.
* Fix repr error of {class}`nk.sampler.rules.MultipleRules` [#1729](https://github.com/netket/netket/pull/1729).
* Solve an issue with RK Integrators that could not be initialised with integer `t0` initial time if `dt` was a float, as well as a wrong `repr` method leading to uncomprehensible stacktraces [#1736](https://github.com/netket/netket/pull/1736).


## NetKet 3.11.1 (19 february 2024)

Bugfix release to solve two issues:

* Fix `reset_chains=True` does not work in `NETKET_EXPERIMENTAL_SHARDING` mode [#1727](https://github.com/netket/netket/pull/1727).
* Fix unsolvable deprecation warning when using `DoubledHilbert` [#1728](https://github.com/netket/netket/pull/1728).


## NetKet 3.11 (~💘 16 february 2024)

This release supports Python 3.12 through the latest release of Numba, introduces several new jax-compatible operators and adds a new experimental way to distribute calculations among multiple GPUs without using MPI.

We have a few breaking changes as well: deprecations that were issued more than 18 months ago have now been finalized, most notable the `dtype` argument to several models and layers, some keywords to GCNN and setting the number of chains of exact samplers.

### New Features

* Recurrent neural networks and layers have been added to `nkx.models` and `nkx.nn` [#1305](https://github.com/netket/netket/pull/1305).
* Added experimental support for running NetKet on multiple jax devices (as an alternative to MPI). It is enabled by setting the environment variable/configuration flag `NETKET_EXPERIMENTAL_SHARDING=1`. Parallelization is achieved by distributing the Markov chains / samples equally across all available devices utilizing [`jax.Array` sharding](https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html). On GPU multi-node setups are supported via [jax.distribued](https://jax.readthedocs.io/en/latest/multi_process.html), whereas on CPU it is limited to a single process but several threads can be used by setting `XLA_FLAGS='--xla_force_host_platform_device_count=XX'` [#1511](https://github.com/netket/netket/pull/1511).
* {class}`netket.experimental.operator.FermionOperator2nd` is a new Jax-compatible implementation of fermionic operators. It can also be constructed starting from a standard fermionic operator by calling `operator.to_jax_operator()`, or used in combination with `pyscf` converters[#1675](https://github.com/netket/netket/pull/1675),[#1684](https://github.com/netket/netket/pull/1684).
* {class}`netket.operator.LocalOperatorJax` is a new Jax-compatible implementation of local operators. It can also be constructed starting from a standard operator by calling `operator.to_jax_operator()` [#1654](https://github.com/netket/netket/pull/1654).
* The logger interface has been formalised and documented in the abstract base class {class}`netket.logging.AbstractLog` [#1665](https://github.com/netket/netket/pull/1665).
* The {class}`~netket.experimental.sampler.ParticleExchange` sampler and corresponding rule {class}`~netket.experimental.sampler.rules.ParticleExchangeRule` has been added, which special cases {class}`~netket.sampler.ExchangeSampler` to fermionic spaces in order to avoid proposing moves where the two site exchanged have the same population [#1683](https://github.com/netket/netket/issues/1683).

### Breaking Changes

* The {class}`netket.models.Jastrow` wave-function now only has {math}`N (N-1)` variational parameters, instead of the {math}`N^2` redundant ones it had before. Saving and loading format has now changed and won't be compatible with previous versions[#1664](https://github.com/netket/netket/pull/1664).
* Finalize deprecations of some old methods in `nk.sampler` namespace (see original commit [1f77ad8267e16fe8b2b2641d1d48a0e7ae94832e](https://github.com/netket/netket/commit/1f77ad8267e16fe8b2b2641d1d48a0e7ae94832e))
* Finalize deprecations of 2D input to DenseSymm layers, which now turn into error and `extra_bias` option of Equivariant Networks/GCNNs (see original commit [c61ea542e9d0f3e899d87a7471dea96d4f6b152d](https://github.com/netket/netket/commit/c61ea542e9d0f3e899d87a7471dea96d4f6b152d))
* Finalize deprecations of very old input/properties to Lattices [0f6f520da9cb6afcd2361dd6fd029e7ad6a2693e](https://github.com/netket/netket/commit/0f6f520da9cb6afcd2361dd6fd029e7ad6a2693e))
* Finalie the deprecation for `dtype=` attribute of several modules in `nk.nn` and `nk.models`, which has been printing an error since April 2022. You should update usages of `dtype=` to `param_dtype=` [#1724](https://github.com/netket/netket/issues/1724)


### Deprecations

* `MetropolisSampler.n_sweeps` has been renamed to {attr}`~netket.sampler.MetropolisSampler.MetropolisSampler.sweep_size` for clarity. Using `n_sweeps` when constructing the sampler now throws a deprecation warning; `sweep_size` should be used instead going forward [#1657](https://github.com/netket/netket/issues/1657).
* Samplers and metropolis rules defined as {func}`netket.utils.struct.dataclass` are deprecated because the base class is now a {class}`netket.utils.struct.Pytree`. The only change needed is to remove the dataclass decorator and define a standard init method [#1653](https://github.com/netket/netket/issues/1653).
* The `out` keyword of Discrete Hilbert indexing methods (`all_states`, `numbers_to_states` and `states_to_numbers`) is deprecated and will be removed in the next release. Plan ahead and remove usages to avoid breaking your code 3 months from now [#1725](https://github.com/netket/netket/issues/1725)!

### Internal changes
* A new class {class}`netket.utils.struct.Pytree`, can be used to create Pytrees for which inheritance autoamtically works and for which it is possible to define `__init__`. Several structures such as samplers and rules have been transitioned to this new interface instead of old style `@struct.dataclass` [#1653](https://github.com/netket/netket/issues/1653).
* The {class}`~netket.experimental.operator.FermionOperator2nd` and related classes now store the constant diagonal shift as another term instead of a completely special cased scalar value. The same operators now also respect the `cutoff` keyword argument more strictly [#1686](https://github.com/netket/netket/issues/1686).
* Dtypes of the matrix elements of operators are now handled more correctly, and fewer warnings are raised when running NetKet in X32 mode. Moreover, operators like Ising now default to floating point dtype even if the coefficients are integers [#1697](https://github.com/netket/netket/issues/1697).

### Bug Fixes
* Support multiplication of Discrete Operators by Sparse arrays [#1661](https://github.com/netket/netket/issues/1661).

## NetKet 3.10.2 (14 november 2023)

### Bug Fixes

* Fixed a bug where it was not possible to recompile functions using two identical but different instances of PauliStringJax [#1647](https://github.com/netket/netket/pull/1647).
* Fixed a minor bug where chunking was never actually used inside of {meth}`~netket.vqs.MCState.local_estimators`. This will turn on chunking for some other drivers such as {class}`netket.experimental.driver.VMC_SRt` and {class}`netket.experimental.driver.TDVPSchmitt`) [#1650](https://github.com/netket/netket/pull/1650).
* {class}`netket.operator.Ising` now throws an error when it is constructed using a non-{class}`netket.hilbert.Spin` hilbert space [#1648](https://github.com/netket/netket/pull/1648).

## NetKet 3.10.1 (8 november 2023)

### Bug Fixes
* Added support for neural networks with complex parameters to {class}`netket.experimental.driver.VMC_SRt`, which was just crashing with unreadable errors before [#1644](https://github.com/netket/netket/pull/1644).

## NetKet 3.10 (🥶 7 november 2023)

The highlights of this version are a new experimental driver to optimise networks with millions of parameters using SR, and introduces new utility functions to convert a pyscf molecule to a netket Hamiltonian.

Read below for a more detailed changelog

### New Features

* Added new {class}`netket.experimental.driver.VMC_SRt` driver, which leads in identical parameter updates as the standard Stochastic Reconfiguration with diagonal shift regularization. Therefore, it is essentially equivalent to using the standard {class}`netket.driver.VMC` with the {class}`netket.optimizer.SR` preconditioner. The advantage of this method is that it requires the inversion of a matrix with side number of samples instead of number of parameters, making this formulation particularly useful in typical deep learning scenarios [#1623](https://github.com/netket/netket/pull/1623).
* Added a new function {func}`netket.experimental.operator.from_pyscf_molecule` to construct the electronic hamiltonian of a given molecule specified through pyscf. This is accompanied by {func}`netket.experimental.operator.pyscf.TV_from_pyscf_molecule` to compute the T and V tensors of a pyscf molecule [#1602](https://github.com/netket/netket/pull/1602).
* Added the operator computing the Rényi2 entanglement entropy on Hilbert spaces with discrete dofs [#1591](https://github.com/netket/netket/pull/1591).
* It is now possible to disable netket's double precision default activation and force all calculations to be performed using single precision by setting the environment variable/configuration flag `NETKET_ENABLE_X64=0`, which also sets `JAX_ENABLE_X64=0`. When running with this flag, the number of warnings printed by jax is considerably reduced as well [#1544](https://github.com/netket/netket/pull/1544).
* Added new shortcuts to build the identity operator as {func}`netket.operator.spin.identity` and {func}`netket.operator.boson.identity` [#1601](https://github.com/netket/netket/pull/1601).
* Added new {class}`netket.hilbert.Particle` constructor that only takes as input the number of dimensions of the system [#1577](https://github.com/netket/netket/pull/1577).
* Added new {class}`netket.experimental.models.Slater2nd` model implementing a Slater ansatz [#1622](https://github.com/netket/netket/pull/1622).
* Added new {func}`netket.jax.logdet_cmplx` function to compute the complex log-determinant of a batch of matrices [#1622](https://github.com/netket/netket/pull/1622).

### Breaking changes

* {class}`netket.experimental.hilbert.SpinOrbitalFermions` attributes have been changed: {attr}`~netket.experimental.hilbert.SpinOrbitalFermions.n_fermions` now always returns an integer with the total number of fermions in the system (if specified). A new attribute {attr}`~netket.experimental.hilbert.SpinOrbitalFermions.n_fermions_per_spin` has been introduced that returns the same tuple of fermion number per spin subsector as before. A few fields are now marked as read-only as modifications where ignored [#1622](https://github.com/netket/netket/pull/1622).
* The {class}`netket.nn.blocks.SymmExpSum` layer is now normalised by the number of elements in the symmetry group in order to maintain a reasonable normalisation [#1624](https://github.com/netket/netket/pull/1624).
* The labelling of spin sectors in {func}`netket.experimental.operator.fermion.create` and similar operators has now changed from the eigenvalue of the spin operator ({math}`\pm 1/2` and so on) to the eigenvalue of the Pauli matrices ({math}`\pm 1` and so on) [#1637](https://github.com/netket/netket/pull/1637).
* The connected elements and expectation values of all non-simmetric fermionic operators is now changed in order to be correct [#1640](https://github.com/netket/netket/pull/1640).

### Improvements

* Considerably reduced the memory consumption of {class}`~netket.operator.LocalOperator`, especially in the case of large local hilbert spaces. Also leveraged sparsity in the terms to speed up compilation (`_setup`) in the same cases [#1558](https://github.com/netket/netket/pull/1558).
* {class}`netket.nn.blocks.SymmExpSum` now works with inputs of arbitrary dimensions, while previously it errored for all inputs that were not 2D [#1616](https://github.com/netket/netket/pull/1616)
* Stop using `FrozenDict` from `flax` and instead return standard dictionaries for the variational parameters from the variational state. This makes it much easier to edit parameters [#1547](https://github.com/netket/netket/pull/1547).
* Vastly improved, finally readable documentation of all Flax modules and neural network architectures [#1641](https://github.com/netket/netket/pull/1641).

### Bug Fixes

* Fixed minor bug where {class}`netket.operator.LocalOperator` could not be built with `np.matrix` object obtained by converting scipy sparse matrices to dense [#1597](https://github.com/netket/netket/pull/1597).
* Raise correct error instead of unintelligible one when multiplying {class}`netket.experimental.operator.FermionOperator2nd` with other operators [#1599](https://github.com/netket/netket/pull/1599).
* Do not rescale the output of {func}`netket.jax.jacobian` by the square root of number of samples. Previously, when specifying `center=True` we were incorrectly rescaling the output [#1614](https://github.com/netket/netket/pull/1614).
* Fix bug in {class}`netket.operator.PauliStrings` that caused the dtype to get out of sync with the dtype of the internal arrays, causing errors when manipulating them symbolically [#1619](https://github.com/netket/netket/pull/1619).
* Fix bug that prevented the use of {class}`netket.operator.DiscreteJaxOperator` as observables with all drivers [#1625](https://github.com/netket/netket/pull/1625).
* Fermionic operator `get_conn` method was returning values as if the operator was transposed, and has now been fixed. This will break the expectation value of non-simmetric fermionic operators, but hopefully nobody was looking into them [#1640](https://github.com/netket/netket/pull/1640).

## NetKet 3.9.2

This release requires at least Python 3.9 and Jax 0.4.

### Bug Fixes

* Fix a bug introduced in version 3.9 for {class}`netket.experimental.driver.TDVPSchmitt` which resulted in the wrong dynamics [#1551](https://github.com/netket/netket/pull/1551).

## NetKet 3.9.1

### Bug Fixes

* Fix a bug in the construction of {class}`netket.operator.PauliStringsJax` in some cases [#1539](https://github.com/netket/netket/pull/1539).

## NetKet 3.9 (🔥 24 July 2023)

This release requires Python 3.8 and Jax 0.4.

### New Features
* {class}`netket.callbacks.EarlyStopping` now supports relative tolerances for determining when to stop [#1481](https://github.com/netket/netket/pull/1481).
* {class}`netket.callbacks.ConvergenceStopping` has been added, which can stop a driver when the loss function reaches a certain threshold [#1481](https://github.com/netket/netket/pull/1481).
* A new base class {class}`netket.operator.DiscreteJaxOperator` has been added, which will be used as a base class for a set of operators that are jax-compatible [#1506](https://github.com/netket/netket/pull/1506).
* {func}`netket.sampler.rules.HamiltonianRule` has been split into two implementations, {class}`netket.sampler.rules.HamiltonianRuleJax` and {class}`netket.sampler.rules.HamiltonianRuleNumba`, which are to be used for {class}`~netket.operator.DiscreteJaxOperator` and standard numba-based {class}`~netket.operator.DiscreteOperator`s. The user-facing API is unchanged, but the returned type might now depend on the input operator [#1514](https://github.com/netket/netket/pull/1514).
* {class}`netket.operator.PauliStringsJax` is a new operator that behaves as {class}`netket.operator.PauliStrings` but is Jax-compatible, meaning that it can be used inside of jax-jitted contexts and works better with chunking. It can also be constructed starting from a standard Ising operator by calling `operator.to_jax_operator()` [#1506](https://github.com/netket/netket/pull/1506).
* {class}`netket.operator.IsingJax` is a new operator that behaves as `netket.operator.Ising` but is Jax-compatible, meaning that it can be used inside of jax-jitted contexts and works better with chunking. It can also be constructed starting from a standard Ising operator by calling `operator.to_jax_operator()` [#1506](https://github.com/netket/netket/pull/1506).
* Added a new method {meth}`netket.operator.LocalOperator.to_pauli_strings` to convert {class}`netket.operator.LocalOperator` to {class}`netket.operator.PauliStrings`. As PauliStrings can be converted to Jax-operators, this now allows to convert arbitrary operators to Jax-compatible ones [#1515](https://github.com/netket/netket/pull/1515).
* The constructor of {meth}`~netket.optimizer.qgt.QGTOnTheFly` now takes an optional boolean argument `holomorphic : Optional[bool]` in line with the other geometric tensor implementations. This flag does not affect the computation algorithm, but will be used to raise an error if the user attempts to call {meth}`~netket.optimizer.qgt.QGTOnTheFly.to_dense()` with a non-holomorphic ansatz. While this might break past code, the numerical results were incorrect.

### Breaking Changes
* The first two axes in the output of the samplers have been swapped, samples are now of shape `(n_chains, n_samples_per_chain, ...)` consistent with `netket.stats.statistics`. Custom samplers need to be updated to return arrays of shape `(n_chains, n_samples_per_chain, ...)` instead of `(n_samples_per_chain, n_chains, ...)`. [#1502](https://github.com/netket/netket/pull/1502)
* The tolerance arguments of {class}`~netket.experimental.dynamics.TDVPSchmitt` have all been renamed to more understandable quantities without inspecting the source code. In particular,  `num_tol` has been renamed to `rcond`, `svd_tol` to `rcond_smooth` and `noise_tol` to `noise_atol`.

### Deprecations
* `netket.vqs.ExactState` has been renamed to {class}`netket.vqs.FullSumState` to better reflect what it does. Using the old name will now raise a warning [#1477](https://github.com/netket/netket/pull/1477).


### Known Issues
* The new `Jax`-friendly operators do not work with {class}`netket.vqs.FullSumState` because they are not hashable. This will be fixed in a minor patch (coming soon).


## NetKet 3.8 (8 May 2023)

This is the last NetKet release to support Python 3.7 and Jax 0.3.
Starting with NetKet 3.9 we will require Jax 0.4, which in turns requires Python 3.8 (and soon 3.9).

### New features
* {class}`netket.hilbert.TensorHilbert` has been generalised and now works with both discrete, continuous or a combination of discrete and continuous hilbert spaces [#1437](https://github.com/netket/netket/pull/1437).
* NetKet is now compatible with Numba 0.57 and therefore with Python 3.11 [#1462](https://github.com/netket/netket/pull/1462).
* The new Metropolis sampling transition proposal rules {func}`netket.sampler.rules.MultipleRules` has been added, which can be used to pick from different transition proposals according to a certain probability distribution.
* The new Metropolis sampling transition proposal rules {func}`netket.sampler.rules.TensorRule` has been added, which can be used to combine different transition proposals acting on different subspaces of the Hilbert space together.
* The new Metropolis sampling transition proposal rules {func}`netket.sampler.rules.FixedRule` has been added, which does not change the configuration.

### Deprecations
* The non-public API function to select the default QGT mode for `QGTJacobian`, located at `nk.optimizer.qgt.qgt_jacobian_common.choose_jacobian_mode` has been renamed and made part of the public API of as `nk.jax.jacobian_default_mode`. If you were using this function, please update your codes [#1473](https://github.com/netket/netket/pull/1473).

### Bug Fixes
* Fix issue [#1435](https://github.com/netket/netket/issues/1435), where a 0-tangent originating from integer samples was not correctly handled by {func}`nk.jax.vjp` [#1436](https://github.com/netket/netket/pull/1436).
* Fixed a bug in {class}`netket.sampler.rules.LangevinRule` when setting `chunk_size` [#1465](https://github.com/netket/netket/pull/1465).

### Improvements
* {class}`netket.operator.ContinuousOperator` has been improved and now they correctly test for equality and generate a consistent hash. Moreover, the internal logic of {class}`netket.operator.SumOperator` and {class}`netket.operator.Potential` has been improved, and they lead to less recompilations when constructed again but identical. A few new attributes for those operators have also been exposed [#1440](https://github.com/netket/netket/pull/1440).
* {func}`nk.nn.to_array` accepts an optional keyword argument `chunk_size`, and related methods on variational states now use the chunking specified in the variational state when generating the dense array [#1470](https://github.com/netket/netket/pull/1470).

### Breaking Changes
* Jax version `0.4` is now required, meaning that NetKet no longer works on Python 3.7.


## NetKet 3.7 (💘 13 february 2023)

### New features
* Input and hidden layer masks can now be specified for {class}`netket.models.GCNN` [#1387](https://github.com/netket/netket/pull/1387).
* Support for Jax 0.4 added [#1416](https://github.com/netket/netket/pull/1416).
* Added a continuous space langevin-dynamics transition rule {class}`netket.sampler.rules.LangevinRule` and its corresponding shorthand for constructing the MCMC sampler {func}`netket.sampler.MetropolisAdjustedLangevin` [#1413](https://github.com/netket/netket/pull/1413).
* Added an experimental Quantum State Reconstruction driver at {class}`netket.experimental.QSR` to reconstruct states from data coming from quantum computers or simulators [#1427](https://github.com/netket/netket/pull/1427).
* Added `netket.nn.blocks.SymmExpSum` flax module that symmetrizes a bare neural network module by summing the wave-function over all possible symmetry-permutations given by a certain symmetry group [#1433](https://github.com/netket/netket/pull/1433).

### Breaking Changes
* Parameters of models {class}`netket.models.GCNN` and layers {class}`netket.nn.DenseSymm` and {class}`netket.nn.DenseEquivariant` are stored as an array of shape '[features,in_features,mask_size]'. Masked parameters are now excluded from the model instead of multiplied by zero [#1387](https://github.com/netket/netket/pull/1387).

### Improvements
* The underlying extension API for Autoregressive models that can be used with Ancestral/Autoregressive samplers has been simplified and stabilized and will be documented as part of the public API. For most models, you should now inherit from {class}`netket.models.AbstractARNN` and define the method {meth}`~netket.models.AbstractARNN.conditionals_log_psi`. For additional performance, implementers can also redefine {meth}`~netket.models.AbstractARNN.__call__` and {meth}`~netket.models.AbstractARNN.conditional` but this should not be needed in general. This will cause some breaking changes if you were relying on the old undocumented interface [#1361](https://github.com/netket/netket/pull/1361).
* {class}`netket.operator.PauliStrings` now works with non-homogeneous Hilbert spaces, such as those obtained by taking the tensor product of multiple Hilbert spaces [#1411](https://github.com/netket/netket/pull/1411).
* The {class}`netket.operator.LocalOperator` now keep sparse matrices sparse, leading to faster algebraic manipulations of those objects. The overall computational and memory cost is, however, equivalent, when running VMC calculations. All pre-constructed operators such as {func}`netket.operator.spin.sigmax` and {func}`netket.operator.boson.create` now build sparse-operators [#1422](https://github.com/netket/netket/pull/1422).
* When multiplying an operator by it's conjugate transpose NetKet does not return anymore a lazy {class}`~netket.operator.Squared` object if the operator is hermitian. This avoids checking if the object is hermitian which greatly speeds up algebric manipulations of operators, and returns more unbiased epectation values [#1423](https://github.com/netket/netket/pull/1423).

### Bug Fixes
* Fixed a bug where {meth}`nk.hilbert.Particle.random_state` could not be jit-compiled, and therefore could not be used in the sampling [#1401](https://github.com/netket/netket/pull/1401).
* Fixed bug [#1405](https://github.com/netket/netket/pull/1405) where {meth}`nk.nn.DenseSymm` and {meth}`nk.models.GCNN` did not work or correctly consider masks [#1428](https://github.com/netket/netket/pull/1428).

### Deprecations
* {meth}`netket.models.AbstractARNN._conditional` has been removed from the API, and its use will throw a deprecation warning. Update your ARNN models accordingly! [#1361](https://github.com/netket/netket/pull/1361).
* Several undocumented internal methods from {class}`netket.models.AbstractARNN` have been removed [#1361](https://github.com/netket/netket/pull/1361).


## NetKet 3.6 (🏔️ 6 November 2022)

### New features
* Added a new 'Full statevector' model {class}`netket.models.LogStateVector` that stores the exponentially large state and can be used as an exact ansatz [#1324](https://github.com/netket/netket/pull/1324).
* Added a new experimental {class}`~netket.experimental.driver.TDVPSchmitt` driver, implementing the signal-to-noise ratio TDVP regularisation by Schmitt and Heyl [#1306](https://github.com/netket/netket/pull/1306).
* Added a new experimental {class}`~netket.experimental.driver.TDVPSchmitt` driver, implementing the signal-to-noise ratio TDVP regularisation by Schmitt and Heyl [#1306](https://github.com/netket/netket/pull/1306).
* QGT classes accept a `chunk_size` parameter that overrides the `chunk_size` set by the variational state object [#1347](https://github.com/netket/netket/pull/1347).
* {func}`~netket.optimizer.qgt.QGTJacobianPyTree` and {func}`~netket.optimizer.qgt.QGTJacobianDense` support diagonal entry regularisation with constant and scale-invariant contributions. They accept a new `diag_scale` argument to pass the scale-invariant component [#1352](https://github.com/netket/netket/pull/1352).
* {func}`~netket.optimizer.SR` preconditioner now supports scheduling of the diagonal shift and scale regularisations [#1364](https://github.com/netket/netket/pull/1364).

### Improvements
* {meth}`~netket.vqs.ExactState.expect_and_grad` now returns a {class}`netket.stats.Stats` object that also contains the variance, as {class}`~netket.vqs.MCState` does [#1325](https://github.com/netket/netket/pull/1325).
* Experimental RK solvers now store the error of the last timestep in the integrator state [#1328](https://github.com/netket/netket/pull/1328).
* {class}`~netket.operator.PauliStrings` can now be constructed by passing a single string, instead of the previous requirement of a list of strings [#1331](https://github.com/netket/netket/pull/1331).
* {class}`~flax.core.frozen_dict.FrozenDict` can now be logged to netket's loggers, meaning that one does no longer need to unfreeze the parameters before logging them [#1338](https://github.com/netket/netket/pull/1338).
* Fermion operators are much more efficient and generate fewer connected elements [#1279](https://github.com/netket/netket/pull/1279).
* NetKet now is completely PEP 621 compliant and does not have anymore a `setup.py` in favour of a `pyproject.toml` based on [hatchling](https://hatch.pypa.io/latest/). To install NetKet you should use a recent version of `pip` or a compatible tool such as poetry/hatch/flint [#1365](https://github.com/netket/netket/pull/1365).
* {func}`~netket.optimizer.qgt.QGTJacobianDense` can now be used with {class}`~netket.vqs.ExactState` [#1358](https://github.com/netket/netket/pull/1358).


### Bug Fixes
* {meth}`netket.vqs.ExactState.expect_and_grad` returned a scalar while {meth}`~netket.vqs.ExactState.expect` returned a {class}`netket.stats.Stats` object with 0 error. The inconsistency has been addressed and now they both return a `Stats` object. This changes the format of the files logged when running `VMC`, which will now store the average under `Mean` instead of `value` [#1325](https://github.com/netket/netket/pull/1325).
* {func}`netket.optimizer.qgt.QGTJacobianDense` now returns the correct output for models with mixed real and complex parameters [#1397](https://github.com/netket/netket/pull/1397)

### Deprecations
* The `rescale_shift` argument of {func}`~netket.optimizer.qgt.QGTJacobianPyTree` and {func}`~netket.optimizer.qgt.QGTJacobianDense` is deprecated in favour the more flexible syntax with `diag_scale`. `rescale_shift=False` should be removed. `rescale_shift=True` should be replaced with `diag_scale=old_diag_shift`. [#1352](https://github.com/netket/netket/pull/1352).
* The call signature of preconditioners passed to {class}`netket.driver.VMC` and other drivers has changed as a consequence of scheduling, and preconditioners should now accept an extra optional argument `step`. The old signature is still supported but is deprecated and will eventually be removed [#1364](https://github.com/netket/netket/pull/1364).


## NetKet 3.5.2 (Bug Fixes) - 30 October 2022

### Bug Fixes
* {class}`~netket.operator.PauliStrings` now support the subtraction operator [#1336](https://github.com/netket/netket/pull/1336).
* Autoregressive networks had a default activation function (`selu`) that did not act on the imaginary part of the inputs. We now changed that, and the activation function is `reim_selu`, which acts independently on the real and imaginary part. This changes nothing for real parameters, but improves the defaults for complex ones [#1371](https://github.com/netket/netket/pull/1371).
* A **major performance degradation** that arose when using {class}`~netket.operator.LocalOperator` has been addressed. The bug caused our operators to be recompiled every time they were queried, imposing a large overhead [1377](https://github.com/netket/netket/pull/1377).


## NetKet 3.5.1 (Bug Fixes)

### New features
* Added a new configuration option `netket.config.netket_experimental_disable_ode_jit` to disable jitting of the ODE solvers. This can be useful to avoid hangs that might happen when working on GPUs with some particular systems [#1304](https://github.com/netket/netket/pull/1304).

### Bug Fixes
* Continuous operatorors now work correctly when `chunk_size != None`. This was broken in v3.5 [#1316](https://github.com/netket/netket/pull/1316).
* Fixed a bug ([#1101](https://github.com/netket/netket/pull/1101)) that crashed NetKet when trying to take the product of two different Hilber spaces. It happened because the logic to build a `TensorHilbert` was ending in an endless loop. [#1321](https://github.com/netket/netket/pull/1321).


## NetKet 3.5 (☀️ 18 August 2022)

[GitHub commits](https://github.com/netket/netket/compare/v3.4...master).

This release adds support and needed functions to run TDVP for neural networks with real/non-holomorphic parameters, an experimental HDF5 logger, and an `MCState` method to compute the local estimators of an observable for a set of samples.

This release also drops support for older version of flax, while adopting the new interface which completely supports complex-valued neural networks. Deprecation warnings might be raised if you were using some layers from `netket.nn` that are now avaiable in flax.

A new, more accurate, estimation of the autocorrelation time has been introduced, but it is disabled by default. We welcome feedback.

### New features

* The method {meth}`~netket.vqs.MCState.local_estimators` has been added, which returns the local estimators `O_loc(s) = 〈s|O|ψ〉 / 〈s|ψ〉` (which are known as local energies if `O` is the Hamiltonian). [#1179](https://github.com/netket/netket/pull/1179)
* The permutation equivariant {class}`nk.models.DeepSetRelDistance` for use with particles in periodic potentials has been added together with an example. [#1199](https://github.com/netket/netket/pull/1199)
* The class {class}`HDF5Log` has been added to the experimental submodule. This logger writes log data and variational state variables into a single HDF5 file. [#1200](https://github.com/netket/netket/issues/1200)
* Added a new method {meth}`~nk.logging.RuntimeLog.serialize` to store the content of the logger to disk [#1255](https://github.com/netket/netket/issues/1255).
* New {class}`nk.callbacks.InvalidLossStopping` which stops optimisation if the loss function reaches a `NaN` value. An optional `patience` argument can be set. [#1259](https://github.com/netket/netket/pull/1259)
* Added a new method {meth}`nk.graph.SpaceGroupBuilder.one_arm_irreps` to construct GCNN projection coefficients to project on single-wave-vector components of irreducible representations. [#1260](https://github.com/netket/netket/issues/1260).
* New method {meth}`~nk.vqs.MCState.expect_and_forces` has been added, which can be used to compute the variational forces generated by an operator, instead of only the (real-valued) gradient of an expectation value. This in general is needed to write the TDVP equation or other similar equations. [#1261](https://github.com/netket/netket/issues/1261)
* TDVP now works for real-parametrized wavefunctions as well as non-holomorphic ones because it makes use of {meth}`~nk.vqs.MCState.expect_and_forces`. [#1261](https://github.com/netket/netket/issues/1261)
* New method {meth}`~nk.utils.group.Permutation.apply_to_id` can be used to apply a permutation (or a permutation group) to one or more lattice indices. [#1293](https://github.com/netket/netket/issues/1293)
* It is now possible to disable MPI by setting the environment variable `NETKET_MPI`. This is useful in cases where mpi4py crashes upon load [#1254](https://github.com/netket/netket/issues/1254).
* The new function {func}`nk.nn.binary_encoding` can be used to encode a set of samples according to the binary shape defined by an Hilbert space. It should be used similarly to {func}`flax.linen.one_hot` and works with non homogeneous Hilbert spaces [#1209](https://github.com/netket/netket/issues/1209).
* A new method to estimate the correlation time in Markov chain Monte Carlo (MCMC) sampling has been added to the {func}`nk.stats.statistics` function, which uses the full FFT transform of the input data. The new method is not enabled by default, but can be turned on by setting the `NETKET_EXPERIMENTAL_FFT_AUTOCORRELATION` environment variable to `1`. In the future we might turn this on by default [#1150](https://github.com/netket/netket/issues/1150).

### Dependencies
* NetKet now requires at least Flax v0.5

### Deprecations

* `nk.nn.Module` and `nk.nn.compact` have been deprecated. Please use the {class}`flax.linen.Module` and {func}`flax.linen.compact` instead.
* `nk.nn.Dense(dtype=mydtype)` and related Modules (`Conv`, `DenseGeneral` and `ConvGeneral`) are deprecated. Please use `flax.linen.***(param_dtype=mydtype)` instead. Before flax v0.5 they did not support complex numbers properly within their modules, but starting with flax 0.5 they now do so we have removed our linear module wrappers and encourage you to use them. Please notice that the `dtype` argument previously used by netket should be changed to `param_dtype` to maintain the same effect. [#...](https://github.com/netket/netket/pull/...)

### Bug Fixes
* Fixed bug where a `nk.operator.LocalOperator` representing the identity would lead to a crash. [#1197](https://github.com/netket/netket/pull/1197)
* Fix a bug where Fermionic operators {class}`nkx.operator.FermionOperator2nd` would not result hermitian even if they were. [#1233](https://github.com/netket/netket/pull/1233)
* Fix serialization of some arrays with complex dtype in `RuntimeLog` and `JsonLog` [#1258](https://github.com/netket/netket/pull/1258)
* Fixed bug where the {class}`nk.callbacks.EarlyStopping` callback would not work as intended when hitting a local minima. [#1238](https://github.com/netket/netket/pull/1238)
* `chunk_size` and the random seed of Monte Carlo variational states are now serialised. States serialised previous to this change can no longer be unserialised [#1247](https://github.com/netket/netket/pull/1247)
* Continuous-space hamiltonians now work correctly with neural networks with complex parameters [#1273](https://github.com/netket/netket/pull/1273).
* NetKet now works under MPI with recent versions of jax (>=0.3.15) [#1291](https://github.com/netket/netket/pull/1291).

## NetKet 3.4.2 (BugFixes & DepWarns again)

[GitHub commits](https://github.com/netket/netket/compare/v3.4.1...v3.4.2).

### Internal Changes
* Several deprecation warnings related to `jax.experimental.loops` being deprecated have been resolved by changing those calls to {func}`jax.lax.fori_loop`. Jax should feel more tranquillo now. [#1172](https://github.com/netket/netket/pull/1172)

### Bug Fixes
* Several _type promotion_ bugs that would end up promoting single-precision models to double-precision have been squashed. Those involved `nk.operator.Ising` and `nk.operator.BoseHubbard`[#1180](https://github.com/netket/netket/pull/1180), `nkx.TDVP` [#1186](https://github.com/netket/netket/pull/1186) and continuous-space samplers and operators [#1187](https://github.com/netket/netket/pull/1187).
* `nk.operator.Ising`, `nk.operator.BoseHubbard` and `nk.operator.LocalLiouvillian` now return connected samples with the same precision (`dtype`) as the input samples. This allows to preserve low precision along the computation when using those operators.[#1180](https://github.com/netket/netket/pull/1180)
* `nkx.TDVP` now updates the expectation value displayed in the progress bar at every time step. [#1182](https://github.com/netket/netket/pull/1182)
* Fixed bug [#1192](https://github.com/netket/netket/pull/1192) that affected most operators (`nk.operator.LocalOperator`) constructed on non-homogeneous hilbert spaces. This bug was first introduced in version 3.3.4 and affects all subsequent versions until 3.4.2. [#1193](https://github.com/netket/netket/pull/1193)
* It is now possible to add an operator and it's lazy transpose/hermitian conjugate [#1194](https://github.com/netket/netket/pull/1194)



## NetKet 3.4.1 (BugFixes & DepWarns)

[GitHub commits](https://github.com/netket/netket/compare/v3.4...v3.4.1).

### Internal Changes
* Several deprecation warnings related to `jax.tree_util.tree_multimap` being deprecated have been resolved by changing those calls to `jax.tree_util.tree_map`. Jax should feel more tranquillo now. [#1156](https://github.com/netket/netket/pull/1156)

### Bug Fixes
* ~`TDVP` now supports model with real parameters such as `RBMModPhase`. [#1139](https://github.com/netket/netket/pull/1139)~ (not yet fixed)
* An error is now raised when user attempts to construct a `LocalOperator` with a matrix of the wrong size (bug [#1157](https://github.com/netket/netket/pull/1157). [#1158](https://github.com/netket/netket/pull/1158)
* A bug where `QGTJacobian` could not be used with models in single precision has been addressed (bug [#1153](https://github.com/netket/netket/pull/1153). [#1155](https://github.com/netket/netket/pull/1155)


## NetKet 3.4 (Special 🧱 edition)

[GitHub commits](https://github.com/netket/netket/compare/v3.3...v3.4).

### New features
* `Lattice` supports specifying arbitrary edge content for each unit cell via the kwarg `custom_edges`. A generator for hexagonal lattices with coloured edges is implemented as `nk.graph.KitaevHoneycomb`. `nk.graph.Grid` again supports colouring edges by direction. [#1074](https://github.com/netket/netket/pull/1074)
* Fermionic hilbert space (`nkx.hilbert.SpinOrbitalFermions`) and fermionic operators (`nkx.operator.fermion`) to treat systems with a finite number of Orbitals have been added to the experimental submodule. The operators are also integrated with [OpenFermion](https://quantumai.google/openfermion). Those functionalities are still in development and we would welcome feedback. [#1090](https://github.com/netket/netket/pull/1090)
* It is now possible to change the integrator of a `TDVP` object without reconstructing it. [#1123](https://github.com/netket/netket/pull/1123)
* A `nk.nn.blocks` has been added and contains an `MLP` (Multi-Layer Perceptron). [#1295](https://github.com/netket/netket/pull/1295)

### Breaking Changes
* The gradient for models with real-parameter is now multiplied by 2. If your model had real parameters you might need to change the learning rate and halve it. Conceptually this is a bug-fix, as the value returned before was wrong (see Bug Fixes section below for additional details) [#1069](https://github.com/netket/netket/pull/1069)
* In the statistics returned by `netket.stats.statistics`, the `.R_hat` diagnostic has been updated to be able to detect non-stationary chains via the split-Rhat diagnostic (see, e.g., Gelman et al., [Bayesian Data Analysis](http://www.stat.columbia.edu/~gelman/book/), 3rd edition). This changes (generally increases) the numerical values of `R_hat` for existing simulations, but should strictly improve its capabilities to detect MCMC convergence failure. [#1138](https://github.com/netket/netket/pull/1138)

### Internal Changes

### Bug Fixes
* The gradient obtained with `VarState.expect_and_grad` for models with real-parameters was off by a factor of $ 1/2 $ from the correct value. This has now been corrected. As a consequence, the correct gradient for real-parameter models is equal to the old times 2. If your model had real parameters you might need to change the learning rate and halve it. [#1069](https://github.com/netket/netket/pull/1069)
* Support for coloured edges in `nk.graph.Grid`, removed in [#724](https://github.com/netket/netket/pull/724), is now restored. [#1074](https://github.com/netket/netket/pull/1074)
* Fixed bug that prevented calling `.quantum_geometric_tensor` on `netket.vqs.ExactState`. [#1108](https://github.com/netket/netket/pull/1108)
* Fixed bug where the gradient of `C->C` models (complex parameters, complex output) was computed incorrectly with `nk.vqs.ExactState`. [#1110](https://github.com/netket/netket/pull/1110)
* Fixed bug where `QGTJacobianDense.state` and `QGTJacobianPyTree.state` would not correctly transform the starting point `x0` if `holomorphic=False`. [#1115](https://github.com/netket/netket/pull/1115)
* The gradient of the expectation value obtained with `VarState.expect_and_grad` for `SquaredOperator`s was off by a factor of 2 in some cases, and wrong in others. This has now been fixed. [#1065](https://github.com/netket/netket/pull/1065).

## NetKet 3.3.2 (🐛 Bug Fixes)

### Internal Changes
* Support for Python 3.10 [#952](https://github.com/netket/netket/pull/952).
* The minimum [optax](https://github.com/deepmind/optax) version is now `0.1.1`, which finally correctly supports complex numbers. The internal implementation of Adam which was introduced in 3.3 ([#1069](https://github.com/netket/netket/pull/1069)) has been removed. If an older version of `optax` is detected, an import error is thrown to avoid providing wrong numerical results. Please update your optax version! [#1097](https://github.com/netket/netket/pull/1097)

### Bug Fixes
* Allow `LazyOperator@densevector` for operators such as lazy `Adjoint`, `Transpose` and `Squared`. [#1068](https://github.com/netket/netket/pull/1068)
* The logic to update the progress bar in {class}`nk.experimental.TDVP` has been improved, and it should now display updates even if there are very sparse `save_steps`. [#1084](https://github.com/netket/netket/pull/1084)
* The `nk.logging.TensorBoardLog` is now lazily initialized to better work in an MPI environment. [#1086](https://github.com/netket/netket/pull/1086)
* Converting a `nk.operator.BoseHubbard` to a `nk.operator.LocalOperator` multiplied by 2 the nonlinearity `U`. This has now been fixed. [#1102](https://github.com/netket/netket/pull/1102)


## NetKet 3.3.1 (🐛 Bug Fixes)

[GitHub commits](https://github.com/netket/netket/compare/v3.3...v3.3.1).

* Initialisation of all implementations of `DenseSymm`, `DenseEquivariant`, `GCNN` now defaults to truncated normals with Lecun variance scaling. For layers without masking, there should be no noticeable change in behaviour. For masked layers, the same variance scaling now works correctly. [#1045](https://github.com/netket/netket/pull/1045)
* Fix bug that prevented gradients of non-hermitian operators to be computed. The feature is still marked as experimental but will now run (we do not guarantee that results are correct). [#1053](https://github.com/netket/netket/pull/1053)
* Common lattice constructors such as `Honeycomb` now accepts the same keyword arguments as `Lattice`. [#1046](https://github.com/netket/netket/pull/1046)
* Multiplying a `QGTOnTheFly` representing the real part of the QGT (showing up when the ansatz has real parameters) with a complex vector now throws an error. Previously the result would be wrong, as the imaginary part [was casted away](https://github.com/netket/netket/issues/789#issuecomment-871145119). [#885](https://github.com/netket/netket/pull/885)


## NetKet 3.3 (🎁 20 December 2021)

[GitHub commits](https://github.com/netket/netket/compare/v3.2...v3.3).

### New features
* The interface to define expectation and gradient function of arbitrary custom operators is now stable. If you want to define it for a standard operator that can be written as an average of local expectation terms, you can now define a dispatch rule for {func}`netket.vqs.get_local_kernel_arguments` and {func}`netket.vqs.get_local_kernel`. The old mechanism is still supported, but we encourage to use the new mechanism as it is more terse. [#954](https://github.com/netket/netket/pull/954)
* {func}`nk.optimizer.Adam` now supports complex parameters, and you can use {func}`nk.optimizer.split_complex` to make optimizers process complex parameters as if they are pairs of real parameters. [#1009](https://github.com/netket/netket/pull/1009)
* Chunking of `MCState.expect` and `MCState.expect_and_grad` computations is now supported, which allows to bound the memory cost in exchange of a minor increase in computation time. [#1006](https://github.com/netket/netket/pull/1006) (and discussions in [#918](https://github.com/netket/netket/pull/918) and [#830](https://github.com/netket/netket/pull/830))
* A new variational state that performs exact summation over the whole Hilbert space has been added. It can be constructed with {class}`nk.vqs.ExactState` and supports the same Jax neural networks as {class}`nk.vqs.MCState`. [#953](https://github.com/netket/netket/pull/953)
* {func}`nk.nn.DenseSymm` allows multiple input features. [#1030](https://github.com/netket/netket/pull/1030)
* [Experimental] A new time-evolution driver {class}`nk.experimental.TDVP` using the time-dependent variational principle (TDVP) has been added. It works with time-independent and time-dependent Hamiltonians and Liouvillians. [#1012](https://github.com/netket/netket/pull/1012)
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
* The {func}`~nk.hilbert.random.flip_state` method used by `MetropolisLocal` now throws an error when called on a {class}`nk.hilbert.ContinuousHilbert` hilbert space instead of entering an endless loop. [#1014](https://github.com/netket/netket/pull/1014)
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
* Subclasses of {class}`netket.models.AbstractARNN` must define the field `machine_pow` [#940](https://github.com/netket/netket/pull/940)
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

* The {py:mod}`netket.utils.group` submodule provides utilities for geometrical and permutation groups. `Lattice` (and its specialisations like `Grid`) use these to automatically construct the space groups of lattices, as well as their character tables for generating wave functions with broken symmetry. [#724](https://github.com/netket/netket/pull/724)
* Autoregressive neural networks, sampler, and masked linear layers have been added to `models`, `sampler` and `nn` [#705](https://github.com/netket/netket/pull/705).


### Breaking Changes

* The `netket.graph.Grid` class has been removed. {ref}`netket.graph.Grid` will now return an instance of {class}`graph.Lattice` supporting the same API but with new functionalities related to spatial symmetries. The `color_edges` optional keyword argument has been removed without deprecation. [#724](https://github.com/netket/netket/pull/724)
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
  allows us to be consistent with the interface of {class}`variational.MCState`, where `n_samples` is the total number of samples
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
* Fix bug [#635](https://github.com/netket/netket/pull/635) where the {class}`graph.Lattice` was not correctly computing neighbours because of floating point issues. [#633](https://github.com/netket/netket/pull/633)
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

  * Some models (previously machines) such as the RBM (Restricted Boltzmann Machine) Machine, NDM (Neural Density Matrix) or MPS (Matrix Product State ansatz) are available in [`Pre-built models`](netket_models_api).

  * Machines, now called models, should be written using [Flax](https://flax.readthedocs.io/en/latest) or another jax framework.

  * Serialization and deserialization functionality has now been moved to {class}`netket.variational.MCState`, which support the standard Flax interface through MsgPack. See [Flax docs](https://flax.readthedocs.io/en/latest/flax.serialization.html) for more information

  * {code}`AbstractMachine.init_random_parameters` functionality has now been absorbed into {meth}`netket.vqs.VariationalState.init_parameters`, which however has a different syntax.

* {ref}`Samplers <Sampler>` now require the Hilbert space upon which they sample to be passed in to the constructor.
Also note that several keyword arguments of the samplers have changed, and new one are available.

* It's now possible to change {ref}`Samplers <Sampler>` dtype, which controls the type of the output. By default they use double-precision samples (`np.float64`). Be wary of type promotion issues with your models.

* {ref}`Samplers <Sampler>` no longer take a machine as an argument.

* {ref}`Samplers <Sampler>` are now immutable (frozen) `dataclasses` (defined through `flax.struct.dataclass`) that only hold the sampling parameters. As a consequence it is no longer possible to change their settings such as `n_chains` or `n_sweeps` without creating a new sampler. If you wish to update only one parameter, it is possible to construct the new sampler with the updated value by using the `sampler.replace(parameter=new_value)` function.

* {ref}`Samplers <Sampler>` are no longer stateful objects. Instead, they can construct an immutable state object `netket.sampler.init_state`, which can be passed to sampling functions such as `netket.sampler.sample`, which now return also the updated state. However, unless you have particular use-cases we advise you use the variational state `MCState` instead.

* The {ref}`netket.optimizer` module has been overhauled, and now only re-exports flax optim module. We advise not to use netket's optimizer but instead to use [optax](https://github.com/deepmind/optax>) .

* The {ref}`netket.optimizer.SR` object now is only a set of options used to compute the SR matrix. The SR matrix, now called `quantum_geometric_tensor` can be obtained by calling {meth}`variational.MCState.quantum_geometric_tensor`. Depending on the settings, this can be a lazy object.

* `netket.Vmc` has been renamed to {class}`netket.VMC`

* {class}`netket.models.RBM` replaces the old {code}`RBM` machine, but has real parameters by default.

* As we rely on Jax, using {code}`dtype=float` or {code}`dtype=complex`, which are weak types, will sometimes lead to loss of precision because they might be converted to `float32`. Use {code}`np.float64` or {code}`np.complex128` instead if you want double precision when defining your models.
