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
from collections.abc import Callable
from collections.abc import Hashable

import jax
import jax.numpy as jnp

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray, struct


class PotentialEnergy(ContinuousOperator):
    r"""Returns the local potential energy defined in afun"""

    _potential_fun: Callable = struct.static_field()
    _coefficient: Array

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        dtype: DType | None = None,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function of x
            coefficient: A coefficient for the ContinuousOperator object
            dtype: Data type of the coefficient
        """

        self._potential_fun = afun
        self._coefficient = jnp.array(coefficient, dtype=dtype)

        super().__init__(hilbert, self._coefficient.dtype)

    @property
    def coefficient(self) -> Array:
        return self._coefficient

    @property
    def is_hermitian(self) -> bool:
        return True

    def _expect_kernel(
        self,
        logpsi: Callable,
        params: PyTree,
        x: Array,
    ) -> Array:
        x_dtype = jnp.promote_types(self.dtype, x.dtype)
        return self.coefficient * jax.vmap(self._potential_fun, in_axes=(0,))(
            x.astype(x_dtype)
        )

    @struct.property_cached(pytree_ignore=True)
    def _attrs(self) -> tuple[Hashable, ...]:
        return (
            self.hilbert,
            self._potential_fun,
            self.dtype,
            HashableArray(self.coefficient),
        )

    def __repr__(self):
        return (
            f"Potential(coefficient={self.coefficient}, function={self._potential_fun})"
        )
