(netket_utils_api)=
# netket.utils

```{eval-rst}
.. currentmodule:: netket.utils

```
## Generic functions

Utility functions and classes.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   is_probably_holomorphic
   
```

## Dataclass and PyTree utilities

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils/struct
   :nosignatures:

   struct.Pytree
   struct.field
   struct.ShardedFieldSpec
```


## Utils

Utility functions and classes.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   HashableArray
   StaticRange
   numbers.StaticZero
   
```

## Tree traversal helpers

NetKet logging backends share a path-aware tree walker implemented in
`netket.utils.tree_walk.walk_tree_with_path`.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   walk_tree_with_path
```

It traverses nested `dict`, `list`, `tuple`, namedtuple, `to_dict()`, and
`to_compound()` structures while threading a path accumulator down to each leaf.
This helper is used internally by the HDF5 and TensorBoard loggers to keep their
tree traversal logic consistent without duplicating the walk itself.

The function docstring includes a minimal example showing how to define
`visit_leaf`, `enter_node`, and `expand_node` callbacks.


## Symmetry Group Manipulation

The group-manipulation API is documented under {ref}`netket_symmetry_api`.
The legacy `netket.utils.group` namespace re-exports the same public objects,
including {class}`~netket.symmetry.group.Permutation`,
{class}`~netket.symmetry.group.PermutationGroup`,
{class}`~netket.symmetry.group.FiniteGroup`, and
{class}`~netket.symmetry.group.PointGroup`.

## History and time-series data

Utilities for storing and managing time-series data, particularly useful for tracking optimization progress and simulation results.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   history.History
   history.HistoryDict
   history.accum_histories_in_tree
```

## Timing utils

Use those utilities to coarsely profile some netket functions or scopes. The timer here
can be nested and can be used in low-level library functions.


```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   timing.Timer
   timing.timed_scope 
   timing.timed 
```
