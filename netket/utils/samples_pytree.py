import abc
import numpy as np

import jax

from flax import struct


class SampleWrapper(abc.ABC):

    @property
    @abc.abstractmethod
    def _n_batch_dim(self):
        # static number of batch dimensions of the samples
        # subclasses are expected to implement this
        # should be calculated dynacmially as a function of the number of dimension of the data
        # since we want it to increase automatically if samples are e.g. generated inside a vmap
        return NotImplemented

    @property
    def dtype(self):
        # TODO we could return a tree of dtypes here
        return None

    @property
    def ndim(self):
        return self._n_batch_dim + 1

    @property
    def _batch_shape(self):
        # shape of the batch dims; assume the same for all leaves
        return jax.tree.leaves(self)[0].shape[: self._n_batch_dim]

    @property
    def _samp_size(self):
        # sum of all sizes
        return sum(
            int(np.prod(x.shape[self._n_batch_dim :])) for x in jax.tree.leaves(self)
        )

    @property
    def shape(self):
        return self._batch_shape + (self._samp_size,)

    def reshape(self, *shape):
        if len(shape) == 1 and hasattr(shape[0], "len"):
            (shape,) = shape
        # only support reshape of the batch dims
        assert len(shape) >= 1
        if shape[-1] == -1:
            assert np.prod(shape[:-1]) == np.prod(self.shape[:-1])
            shape = shape[:-1] + (self._samp_size,)
        else:
            assert shape[-1] == self._samp_size
        return jax.tree.map(
            lambda x: x.reshape(shape[:-1] + x.shape[self._n_batch_dim :]), self
        )

    def swapaxes(self, axis1, axis2):
        assert axis1 < self._n_batch_dim
        assert axis2 < self._n_batch_dim
        return jax.tree.map(lambda x: x.swapaxes(axis1, axis2), self)

    def __getitem__(self, *idx):
        # only supports indexing of the batch dims
        dummy_leaf = jax.tree.leaves(self)[0]
        n_dim_removed = (
            dummy_leaf.ndim - jax.eval_shape(lambda x: x[*idx], dummy_leaf).ndim
        )
        assert n_dim_removed <= self._n_batch_dim
        return jax.tree.map(lambda x: x[*idx], self)


# example implementation using a tuple of sub states for TensorHilbert
# (would work for any pytree of sub_states)
@struct.dataclass
class SampleWrapperExample(SampleWrapper):
    sub_states: tuple[jax.Array]  # pytree of substates
    # ShapeDtypeStruct's of the sub states without batch dims; in principle we would only need ndim of one of them
    _structure: tuple[jax.ShapeDtypeStruct] = struct.field(pytree_node=False)

    @property
    def _n_batch_dim(self):
        # assume all are consitent
        # TODO could check
        return jax.tree.leaves(
            jax.tree.map(lambda a, b: a.ndim - b.ndim, self.sub_states, self._structure)
        )[0]
