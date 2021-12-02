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
from typing import Union, List, Optional, Callable

from netket.utils.types import DType, PyTree, Array

import functools

from netket.operator import ContinuousOperator

import jax.numpy as jnp


class SumOperator(ContinuousOperator):
    r"""This class implements the action of the _expect_kernel()-method of
    ContinuousOperator for a sum of ContinuousOperator objects.
    """

    def __init__(
        self,
        *operators: List,
        coefficients: Union[float, List[float]] = 1.0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Returns the action of a sum of local operators.
        Args:
            operators: A list of ContinuousOperator objects
            coefficients: A coefficient for each ContinuousOperator object
            dtype: Data type of the matrix elements. Defaults to `np.float64`
        """

        hil = [op.hilbert for op in operators]
        if not all(_ == hil[0] for _ in hil):
            raise NotImplementedError(
                "Cannot add operators on different hilbert spaces"
            )

        self._ops = operators
        self._coeff = coefficients

        if dtype is None:
            dtype = functools.reduce(
                lambda dt, op: jnp.promote_types(dt, op.dtype), operators, float
            )
        self._dtype = dtype

        super().__init__(hil[0], self._dtype)

        self._is_hermitian = all([op.is_hermitian for op in operators])

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            op._expect_kernel(logpsi, params, x, data[i])
            for i, op in enumerate(self._ops)
        ]

        return sum(result)

    def _expect_kernel_batched(
        self, logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            op._expect_kernel_batched(logpsi, params, x, data[i])
            for i, op in enumerate(self._ops)
        ]

        return sum(result)

    def _pack_arguments(self):
        return [self._coeff * jnp.array(op._pack_arguments()) for op in self._ops]

    def __repr__(self):
        return f"SumOperator(coefficients={self._pack_arguments()})"
