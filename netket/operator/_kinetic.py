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
from typing import Optional, Callable, Union, List
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils.types import DType, PyTree, Array
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator


class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: Union[float, List[float]],
        dtype: Optional[DType] = None,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self._mass = jnp.asarray(mass, dtype=dtype)

        self._is_hermitian = np.allclose(self._mass.imag, 0.0)

        super().__init__(hilbert, self._mass.dtype)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, mass: Optional[PyTree]
    ):
        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jax.grad(logpsi_x)

        y, f_jvp = jax.linearize(dlogpsi_x, x)
        basis = jnp.eye(x.shape[0], dtype=y.dtype)

        dp_dx2 = jnp.diag(jax.vmap(f_jvp)(basis))

        dp_dx = dlogpsi_x(x) ** 2

        res = -0.5 * jnp.sum(mass * (dp_dx2 + dp_dx), axis=-1)
        return res

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel_batched(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: Optional[PyTree]
    ):
        return self._expect_kernel(logpsi, params, x, coefficient)

    def _pack_arguments(self) -> PyTree:
        return 1.0 / self._mass

    def __repr__(self):
        return f"KineticEnergy(m={self._mass})"
