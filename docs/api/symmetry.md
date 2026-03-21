(netket_symmetry_api)=
# netket.symmetry

(Work in progress)

```{note}
To construct a symmetrized ansatz of the form log(O|ψ⟩) where O is a symmetry
projector, see {ref}`netket_nn_apply_operator_api` in {func}`netket.vqs.apply_operator`.
```

Module with classes and utilities needed to operate on symmetry groups and their representations.

The theory needed to understand those tools is discussed in {doc}`../advanced/symmetry` and an example notebook is in {doc}`../tutorials/symmetry_tutorial`

## Common Symmetry Representations

```{eval-rst}
.. currentmodule:: netket.symmetry

.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   spin_flip_representation
   canonical_representation
```

## Representation Classes

```{eval-rst}
.. currentmodule:: netket.symmetry

.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   Representation
   LabeledRepresentation
   TranslationRepresentation
```

## Symmetry Group Manipulation

```{eval-rst}
.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   group.FiniteGroup

   group.Permutation
   group.PermutationGroup
   group.cyclic_group

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

## Internal Objects

These classes are returned by methods such as
{meth}`~netket._src.symmetry.labeled_representation.LabeledRepresentation.coset_filter`
but are not part of the public ``netket.symmetry`` namespace.
They are documented here for reference.

```{eval-rst}
.. currentmodule:: netket._src.symmetry

.. autosummary::
   :toctree: _generated/symmetry
   :nosignatures:

   labeled_representation_coset_filter.LabeledRepresentationCosetFilter
   translation_coset_filter.TranslationCosetFilter
```
