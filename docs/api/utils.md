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


## Symmetry Group Manipulation

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   group.Permutation 
   group.PermutationGroup 

   group.FiniteGroup
   group.PointGroup

```

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

