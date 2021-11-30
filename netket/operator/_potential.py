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
from typing import Optional, Callable
from functools import partial

from netket.utils.types import DType, PyTree, Array

from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator

import jax
import jax.numpy as jnp


class PotentialEnergy(ContinuousOperator):
    r"""Returns the local potential energy defined in afun"""

    def __init__(
        self,
        hilbert: AbstractHilbert,
        afun: Callable,
        coefficient: float = 1.0,
        dtype: Optional[DType] = float,
    ):
        r"""
        Args:
            hilbert: The underlying Hilbert space on which the operator is defined
            afun: The potential energy as function of x
            coefficients: A coefficient for the ContinuousOperator object
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        self._afun = afun

        self.coefficient = jnp.array(coefficient, dtype=dtype)

        super().__init__(hilbert, self.coefficient.dtype)

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: Optional[PyTree]
    ):
        return coefficient * self._afun(x)

    @partial(jax.vmap, in_axes=(None, None, None, 0, None))
    def _expect_kernel_batched(
        self, logpsi: Callable, params: PyTree, x: Array, coefficient: Optional[PyTree]
    ):
        return self._expect_kernel(logpsi, params, x, coefficient)

    @property
    def is_hermitian(self):
        return True

    def _pack_arguments(self):
        return self.coefficient

    def __repr__(self):
        return f"Potential(coefficient={self.coefficient}, function+{self._afun})"
