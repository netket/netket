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
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.types import DType, PyTree, Array
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: float | list[float],
        dtype: DType | None = None,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        dtype: Data type of the mass
        """

        self._mass = jnp.asarray(mass, dtype=dtype)

        self._is_hermitian = np.allclose(self._mass.imag, 0.0)
        self.__attrs = None

        super().__init__(hilbert, self._mass.dtype)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel_single(
        self, logpsi: Callable, params: PyTree, x: Array, inverse_mass: PyTree | None
    ):
        dp_dx = jax.jacrev(logpsi, argnums=1)(params, x) ** 2
        dp_dx2 = jnp.diag(jax.hessian(logpsi, argnums=1)(params, x))
        return -0.5 * jnp.sum(inverse_mass * (dp_dx2 + dp_dx), axis=-1)

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: PyTree | None
    ):
        return self._expect_kernel_single(logpsi, params, x, coefficient)

    def _pack_arguments(self) -> PyTree:
        return 1.0 / self._mass

    @property
    def _attrs(self):
        if self.__attrs is None:
            self.__attrs = (self.hilbert, self.dtype, HashableArray(self.mass))
        return self.__attrs

    def __repr__(self):
        return f"KineticEnergy(m={self._mass})"
