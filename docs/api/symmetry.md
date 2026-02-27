(netket_symmetry_api)=
# netket.symmetry

(Work in progress)

```{note}
To construct a symmetrized ansatz of the form log(O|ψ⟩) where O is a symmetry
projector, see {ref}`netket_nn_apply_operator_api` in {mod}`netket.nn.apply_operator`.
```

Module with classes and utilities needed to operate on symmetry groups and their representations.

The theory needed to understand those tools is discussed in {doc}`../advanced/symmetry` and an example notebook is in {doc}`../tutorials/symmetry_tutorial`

```{eval-rst}
.. currentmodule:: netket.symmetry

.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

	Representation
   canonical_representation
```

## Symmetry Group Manipulation

```{eval-rst}
.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   group.FiniteGroup

   group.Permutation 
   group.PermutationGroup 

   group.PGSymmetry
   group.PointGroup
```

## Common Groups

```{eval-rst}
.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   group.axial
   group.planar
   group.cubic
   group.icosa
```


