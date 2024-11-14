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

## MPI-related

Utility functions and classes.

```{eval-rst}
.. autosummary::
   :toctree: _generated/utils
   :nosignatures:

   mpi.available
   mpi.n_nodes
   mpi.rank
   mpi.MPI_jax_comm
   mpi.MPI_py_comm
   
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

