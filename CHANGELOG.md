NetKet release notes for all versions
=====================================

## NetKet 3.0b2 (unreleased)
* [GitHub commits](https://github.com/netket/netket/compare/v3.0b1...master).


New features
------------

* Group Equivariant Neural Networks have been added to `models` [#620](https://github.com/netket/netket/pull/620)

* Permutation invariant RBM and Permutation invariant dense layer have been added to `models` 
  and `nn.linear` [#573](https://github.com/netket/netket/pull/573)

* Add the property `acceptance` to `MetropolisSampler`'s `SamplerState`, computing the
  MPI-enabled acceptance ratio. [#592](https://github.com/netket/netket/pull/592). 

* Add `StateLog`, a new logger that stores the parameters of the model during the 
  optimization in a folder or in a tar file. [#645](https://github.com/netket/netket/pull/645)

* A warning is now issued if netket detects to be running under `mpirun` but MPI dependencies
  are not installed [#631](https://github.com/netket/netket/pull/631)

* `operator.LocalOperator`s now do not return a zero matrix element on the diagonal if the whole
  diagonal is zero. [#623](https://github.com/netket/netket/pull/623).

* `logger.JSONLog` now automatically flushes at every iteration if it does not consume significant
  CPU cycles. [#599](https://github.com/netket/netket/pull/599)

Breaking Changes
----------------

* `MetropolisSampler.reset_chain` has been renamed to `MetropolisSampler.reset_chains`. 
  Likewise in the constructor of all samplers.

* Briefly during development releases `MetropolisSamplerState.acceptance_ratio` returned
  the percentage (not ratio) of acceptance. `acceptance_ratio` is now deprecated in 
  favour of the correct `acceptance`.

* `models.Jastrow` now internally simmetrizes the matrix before computing its value [#644](https://github.com/netket/netket/pull/644)

* `MCState.evaluate` has been renamed to `MCState.log_value` [#632](https://github.com/netket/netket/pull/632)


Bug Fixes
---------

* Fix `BoseHubbard` usage under jax Hamiltonian Sampling [#662](https://github.com/netket/netket/pull/662)
* Fix `SROnTheFly` for `R->C` models with non homogeneous parameters [#661](https://github.com/netket/netket/pull/661)
* Fix MPI Compilation deadlock when computing expectation values [#655](https://github.com/netket/netket/pull/655)
* Fix bug preventing the creation of a `hilbert.Spin` hilbert space with odd sites and even `S`. [#641](https://github.com/netket/netket/pull/641)
* Fix bug [#635](https://github.com/netket/netket/pull/635) preventing the usage of `NumpyMetropolisSampler` with `MCState.expect` [#635](https://github.com/netket/netket/pull/635)
* Fix bug [#635](https://github.com/netket/netket/pull/635) where the `graph.Lattice` was not correctly computing neighbours because of floating point issues. [#633](https://github.com/netket/netket/pull/633)
* Fix bug the Y pauli matrix, which was stored as its conjugate. [#618](https://github.com/netket/netket/pull/618) [#617](https://github.com/netket/netket/pull/617) [#615](https://github.com/netket/netket/pull/615)


## NetKet 3.0b1 (published beta release)

Too many to count