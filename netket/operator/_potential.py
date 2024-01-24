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
from typing import Callable, Optional
from collections.abc import Hashable

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray

import jax
import jax.numpy as jnp


@struct.dataclass
class PotentialOperatorPyTree:
    """Internal class used to pass data from the operator to the jax kernel.

    This is used such that we can pass a PyTree containing some static data.
    We could avoid this if the operator itself was a pytree, but as this is not
    the case we need to pass as a separte object all fields that are used in
    the kernel.

    We could forego this, but then the kernel could not be marked as
    @staticmethod and we would recompile every time we construct a new operator,
    even if it is identical
    """

    potential_fun: Callable = struct.field(pytree_node=False)
    coefficient: Array


class PotentialEnergy(ContinuousOperator):
    r"""Returns the local potential energy defined in afun"""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function of x
            coefficient: A coefficient for the ContinuousOperator object
            dtype: Data type of the coefficient
        """

        self._afun = afun
        self._coefficient = jnp.array(coefficient, dtype=dtype)

        self.__attrs = None

        super().__init__(hilbert, self._coefficient.dtype)

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    @property
    def is_hermitian(self) -> bool:
        return True

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ) -> Array:
        return data.coefficient * jax.vmap(data.potential_fun, in_axes=(0,))(x)

    def _pack_arguments(self) -> PotentialOperatorPyTree:
        return PotentialOperatorPyTree(self._afun, self.coefficient)

    @property
    def _attrs(self) -> tuple[Hashable, ...]:
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self._afun,
                self.dtype,
                HashableArray(self.coefficient),
            )
        return self.__attrs

    def __repr__(self):
        return f"Potential(coefficient={self.coefficient}, function={self._afun})"
