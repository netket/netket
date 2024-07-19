(netket_jax_api)=
# netket.jax

```{eval-rst}
.. currentmodule:: netket.jax

```

This module contains some *internal utilities* to work with Jax.
This part of the API is not public, and can change without notice.

## Utility functions

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  HashablePartial
  PRNGKey
  PRNGSeq
  mpi_split

```

## Tree Linear Algebra

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  tree_dot
  tree_norm
  tree_ax
  tree_axpy
  tree_cast
  tree_conj
  tree_size
  tree_leaf_iscomplex
  tree_ishomogeneous
  tree_ravel
  tree_to_real
```

## Dtype tools

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  dtype_complex
  is_complex_dtype
  maybe_promote_to_complex

```

## Complex-aware AD

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  expect
  vjp
  jacobian
  jacobian_default_mode
```

## Chunked operations

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  chunk
  unchunk
  apply_chunked
  vmap_chunked
  vjp_chunked
```

## Math

```{eval-rst}
.. autosummary::
  :toctree: _generated/jax
  :nosignatures:

  logsumexp_cplx
  logdet_cmplx
```

