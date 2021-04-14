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

from dataclasses import dataclass

import numpy as np

from .types import Array, DType


@dataclass(frozen=True)
class HashableArray:
    """
    This class wraps a numpy or jax array in order to make it hashable and
    equality comparable (which is necessary since a well-defined hashable object
    needs to satisfy :code:`obj1 == obj2` whenever :code:`hash(obj1) == hash(obj2)`.

    The underlying array can also be accessed using :code:`numpy.asarray(self)`.
    """

    wrapped: Array
    """The wrapped array. Note that this array is read-only."""

    def __post_init__(self):
        object.__setattr__(self, "wrapped", self.wrapped.copy())
        if isinstance(self.wrapped, np.ndarray):
            self.wrapped.flags.writeable = False
        object.__setattr__(self, "_HashableArray__hash", hash(self.wrapped.tobytes()))

    def __hash__(self):
        return self.__hash

    def __eq__(self, other):
        return np.all(self.wrapped == other.wrapped)

    def __array__(self, dtype: DType = None):
        if dtype is None:
            dtype = self.wrapped.dtype
        return self.wrapped.__array__(dtype)
