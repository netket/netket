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
from typing import Union, List, Optional, Callable, Any, Tuple

from netket.jax.utils import is_scalar
from netket.utils.types import DType, PyTree, Array

import functools

from netket.operator import ContinuousOperator
from netket.utils import struct, HashableArray

import jax.numpy as jnp


@struct.dataclass
class SumOperatorPyTree:
    ops: Tuple[ContinuousOperator, ...] = struct.field(pytree_node=False)
    coeffs: Array
    op_data: Any


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

        if is_scalar(coefficients):
            coefficients = [coefficients for _ in operators]

        if len(operators) != len(coefficients):
            raise AssertionError("Each operator needs a coefficient")

        new_operators = []
        new_coeffs = []
        for op, c in zip(operators, coefficients):
            if isinstance(op, SumOperator):
                new_operators = new_operators + list(op._ops)
                new_coeffs = new_coeffs + list(c * op._coeff)
            else:
                new_operators.append(op)
                new_coeffs.append(c)

        operators = new_operators
        coefficients = jnp.asarray(new_coeffs, dtype=dtype)

        self._ops = tuple(operators)
        self._coeff = coefficients

        if dtype is None:
            dtype = functools.reduce(
                lambda dt, op: jnp.promote_types(dt, op.dtype), operators, float
            )
        super().__init__(hil[0], dtype)

        self._is_hermitian = all([op.is_hermitian for op in operators])
        self._hash = None
        self.__attrs = None

    @property
    def is_hermitian(self):
        return self._is_hermitian

    @property
    def operators(self):
        return self._ops

    @property
    def coefficients(self):
        return self._coeff

    @staticmethod
    def _expect_kernel(
        logpsi: Callable, params: PyTree, x: Array, data: Optional[PyTree]
    ):
        result = [
            data.coeffs[i] * op._expect_kernel(logpsi, params, x, op_data)
            for i, (op, op_data) in enumerate(zip(data.ops, data.op_data))
        ]

        return sum(result)

    def _pack_arguments(self):
        return SumOperatorPyTree(
            self._ops, self._coeff, tuple(op._pack_arguments() for op in self._ops)
        )

    @property
    def _attrs(self):
        if self.__attrs is None:
            self.__attrs = (
                self.hilbert,
                self._ops,
                self.dtype,
                HashableArray(self._coeff),
            )
        return self.__attrs

    def __repr__(self):
        return (
            f"SumOperator(operators={self.operators}, coefficients={self.coefficients})"
        )
