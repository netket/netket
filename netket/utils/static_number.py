# Copyright 2023 The NetKet Authors - All rights reserved.
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

from numbers import Number

import jax
import jax.numpy as jnp
import numpy as np

from . import struct
from .types import DType


@struct.dataclass
class StaticZero(Number):
    """
    A static representation of 0, which can be used in Jax and attempts to behave like
    a normal jax array.

    Used to be able to test for hard-zeros in jax code
    """

    dtype: DType = struct.field(pytree_node=False, default=bool)

    shape = property(lambda _: ())
    ndim = property(lambda _: 1)
    weak_dtype = property(lambda _: True)
    aval = property(
        lambda self: jax.abstract_arrays.ShapedArray(self.shape, self.dtype)
    )

    __add__ = lambda self, o: self if isinstance(o, StaticZero) else o
    __radd__ = lambda self, o: self if isinstance(o, StaticZero) else o
    __sub__ = lambda self, o: self if isinstance(o, StaticZero) else -o
    __rsub__ = lambda self, o: self if isinstance(o, StaticZero) else o
    __mul__ = lambda self, o: self
    __rmul__ = lambda self, o: self
    __neg__ = lambda self: self
    __bool__ = lambda self: False
    __eq__ = lambda self, o: True if isinstance(o, StaticZero) else False
    astype = lambda self, dtype: StaticZero(dtype)

    def __array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return np.zeros(self.shape, dtype=dtype)

    def __jax_array__(self, dtype=None):
        if dtype is None:
            dtype = self.dtype
        return jnp.zeros(self.shape, dtype=dtype)
