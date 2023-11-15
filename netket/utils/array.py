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

import numpy as np
import jax

from .types import Array, DType, Shape
from .struct import dataclass


@dataclass(cache_hash=True)
class HashableArray:
    """
    This class wraps a numpy or jax array in order to make it hashable and
    equality comparable (which is necessary since a well-defined hashable object
    needs to satisfy :code:`obj1 == obj2` whenever :code:`hash(obj1) == hash(obj2)`.

    The underlying array can also be accessed using :code:`numpy.asarray(self)`.
    """

    wrapped: Array
    """The wrapped array. Note that this array is read-only."""

    def __pre_init__(self, wrapped):
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

        return (wrapped,), {}

    def __hash__(self):
        return hash(self.wrapped.tobytes())

    def __eq__(self, other):
        return type(other) is HashableArray and np.all(self.wrapped == other.wrapped)

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
