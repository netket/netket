# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import numpy as np
import jax

from .types import Array, DType, Shape


class HashableArray:
    """
    This class wraps a numpy or jax array in order to make it hashable and
    equality comparable (which is necessary since a well-defined hashable object
    needs to satisfy :code:`obj1 == obj2` whenever :code:`hash(obj1) == hash(obj2)`.

    The underlying array can also be accessed using :code:`numpy.asarray(self)`.
    """

    def __init__(self, wrapped: Array):
        """
        Wraps an array into an object that is hashable, and that can be
        converted again into an array.

        Forces all arrays to numpy and sets them to readonly.
        They can be converted back to jax later or a writeable numpy copy
        can be created by using `np.array(...)`

        The hash is computed by hashing the whole content of the array.

        Args:
            wrapped: array to be wrapped
        """
        if isinstance(wrapped, HashableArray):
            wrapped = wrapped.wrapped
        else:
            if isinstance(wrapped, jax.Array):
                # __array__ only works if it's a numpy array.
                wrapped = np.array(wrapped)
            else:
                wrapped = wrapped.copy()
            if isinstance(wrapped, np.ndarray):
                wrapped.flags.writeable = False

        self._wrapped: np.array = wrapped
        self._hash: Optional[int] = None

    @property
    def wrapped(self):
        """The read-only wrapped array."""
        return self._wrapped

    def __hash__(self):
        if self._hash is None:
            self._hash = hash(self.wrapped.tobytes())
        return self._hash

    def __eq__(self, other):
        return (
            type(other) is HashableArray
            and self.shape == other.shape
            and self.dtype == other.dtype
            and hash(self) == hash(other)
        )

    def __array__(self, dtype: DType = None):
        if dtype is None:
            dtype = self.wrapped.dtype
        return self.wrapped.__array__(dtype)

    @property
    def dtype(self) -> DType:
        return self.wrapped.dtype

    @property
    def size(self) -> int:
        return self.wrapped.size

    @property
    def ndim(self) -> int:
        return self.wrapped.ndim

    @property
    def shape(self) -> Shape:
        return self.wrapped.shape

    def __repr__(self) -> str:
        return f"HashableArray({self.wrapped},\n shape={self.shape}, dtype={self.dtype}, hash={hash(self)})"

    def __str__(self) -> str:
        return (
            f"HashableArray(shape={self.shape}, dtype={self.dtype}, hash={hash(self)})"
        )


def array_in(x, ys):
    """
    Interpret ys as a list of arrays, and test if x is equal to any y in ys,
    with exactly the same shape but not exactly the same dtype.

    Note:
        In numpy, :code:`x in ys` is equivalent to :code:`any(x == ys)`,
        which is usually not what we intend when :code:`x.size > 1`.
        JAX arrays will raise an error rather than silently compute it.
    """
    x = x.reshape(1, -1)
    ys = ys.reshape(ys.shape[0], -1)
    return (x == ys).all(axis=1).any()
