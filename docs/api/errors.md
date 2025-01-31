(netket_errors_api)=
# netket.errors

```{eval-rst}
.. currentmodule:: netket.errors

```

Netket has the following classes of errors.


## Error classes

```{eval-rst}
.. autosummary::
  :toctree: _generated/errors
  :nosignatures:

  HilbertIndexingDuringTracingError
  HolomorphicUndeclaredWarning
  JaxOperatorSetupDuringTracingError
  JaxOperatorNotConvertibleToNumba
  NonHolomorphicQGTOnTheFlyDenseRepresentationError
  NumbaOperatorGetConnDuringTracingError
  RealQGTComplexDomainError
  UnoptimalSRtWarning
  SymmModuleInvalidInputShape
  ParameterMismatchError
  InitializePeriodicLatticeOnSmallLatticeWarning
```

## Hilbert space errors

Errors arising when working with Hilbert spaces and their constraints:

```{eval-rst}
.. autosummary::
  :toctree: _generated/errors
  :nosignatures:

  UnoptimisedCustomConstraintRandomStateMethodWarning
  UnhashableConstraintError
  InvalidConstraintInterface
```

## PyTree errors

```{eval-rst}
.. autosummary::
  :toctree: _generated/errors
  :nosignatures:

  NetKetPyTreeUndeclaredAttributeAssignmentError
```
