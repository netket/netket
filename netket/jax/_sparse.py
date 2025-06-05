import jax
import jax.numpy as jnp

from jax import Array
from flax import struct

from ._sort import searchsorted


@struct.dataclass
class COOArray:
    """
    simple sparse coo array written in jax (similar to sparse.COO)
    supports arbitrary number of dimensions
    so far only indexing and conversion to/from dense is implemented

    Args:
        _coords: array containing the indices of the nonzero elements
        data: array containing the values of the nonzero elements
        shape: shape of the array
        fill_value: default value of the nonzero elements (0)
    """

    _coords: (
        Array  # needs to be unique and sorted, these are transposed w.r.t sparse.COO
    )
    data: Array
    shape: tuple = struct.field(pytree_node=False, default=())
    fill_value: Array = 0

    def __getitem__(self, *idx):
        assert self.coords.ndim == 2
        assert self.data.ndim == 1
        assert self._coords.shape[0] == self.data.shape[0]
        assert self._coords.shape[1] == len(self.shape)
        # TODO support more indexing modes
        if len(idx) == 1 and isinstance(idx[0], tuple):
            idx = jnp.array(idx[0]).T
            if not idx.shape[-1] == self.ndim:
                raise NotImplementedError
        elif len(idx) == self.ndim:
            idx = jnp.array(idx[0])
            if not idx.shape[-1] == self.ndim:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # TODO implement searchsorted with tuples?
        k = searchsorted(self._coords, idx)
        mask = (self._coords[k] == idx).all(axis=-1)
        res = self.data[k]
        return jax.lax.select(mask, res, jnp.full_like(res, self.fill_value))

    @property
    def coords(self) -> Array:
        # for consistency with sparse
        # TODO implement transposed version?
        return self._coords.T

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def nnz(self) -> int:
        return self.data.size

    def todense(self) -> Array:
        A = jnp.full(self.shape, self.fill_value, self.dtype)
        return A.at[tuple(self.coords)].set(self.data)

    @classmethod
    def fromdense(cls, A: Array) -> "COOArray":
        # assume that nonzero returns a sorted array
        i = jnp.nonzero(A)
        return cls(jnp.array(i).T, A[i], A.shape)
