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

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


@register_pytree_node_class
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
    def potential_fun(self):
        return self._afun

    @property
    def is_hermitian(self) -> bool:
        return True

    def _expect_kernel(self, logpsi: Callable, params: PyTree, x: Array) -> Array:
        return self.coefficient * jax.vmap(self.potential_fun, in_axes=(0,))(x)

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

    def tree_flatten(self):
        data = (self.coefficient,)
        metadata = {
            "hilbert": self.hilbert,
            "potential_fun": self.potential_fun,
            "dtype": self.dtype,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (coeff,) = data
        hi = metadata["hilbert"]
        potential_fun = metadata["potential_fun"]
        dtype = metadata["dtype"]

        op = cls(hi, potential_fun, coefficient=coeff, dtype=dtype)
        return op
