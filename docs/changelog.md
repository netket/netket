```{include} ../CHANGELOG.md
```
## Symmetry overhaul
### New features
* A clear distinction is now made between groups and their representations via the  `Representation` class. 
* The `Representation` class allows the users to define representations of any given group, not just lattice symmetries. 
* The `Representation` class is equipped with a `project` method that projects a variational state onto the subspace associated to an irreducible representation. 
* Netket now supports representations of permutation groups on spin and fermionic Hilbert spaces via the `PermutationOperator` and `PermutationOperatorFermion` classes. The `get_conn_padded` method of `PermutationOperatorFermion` calculates the sign from permuting the occupancies of single-particle states. 
* A new [tutorial](../docs/tutorials/symmetry_tutorial.ipynb) explaining how to use these tools is available. 
* [Documentation](../docs/advanced/symmetry.md) regarding symmetries and representation theory is available