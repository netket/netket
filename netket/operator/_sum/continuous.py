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
from collections.abc import Hashable, Iterable


from netket.utils import HashableArray, struct
from netket.utils.types import DType, PyTree, Array
from netket.operator import ContinuousOperator

from netket.operator._sum.base import SumOperator, SumOperatorMeta


class SumContinuousMeta(struct.pytree.PytreeMeta, SumOperatorMeta):
    """
    Workaround to allow SumContinuousOperator to be a subclass of both
    ContinuousOperator and SumOperator, because the first is a
    Pytree and the latter has the custom dispatch logic.
    """

    pass


class SumContinuousOperator(
    SumOperator, ContinuousOperator, metaclass=SumContinuousMeta
):
    r"""This class implements the action of the _expect_kernel()-method of
    ContinuousOperator for a sum of ContinuousOperator objects.
    """

    _operators: tuple[ContinuousOperator, ...]
    _coefficients: Array
    _is_hermitian: bool = struct.static_field()

    def __init__(
        self,
        *operators: tuple[ContinuousOperator, ...],
        coefficients: float | Iterable[float] = 1.0,
        dtype: DType | None = None,
    ):
        r"""
        Returns the action of a sum of local operators.
        Args:
            operators: A list of ContinuousOperator objects
            coefficients: A coefficient for each ContinuousOperator object
            dtype: Data type of the coefficients
        """
        self._is_hermitian = all([op.is_hermitian for op in operators])
        super().__init__(
            operators, operators[0].hilbert, coefficients=coefficients, dtype=dtype
        )

    @property
    def is_hermitian(self) -> bool:
        return self._is_hermitian

    def _expect_kernel(
        self,
        logpsi: Callable,
        params: PyTree,
        x: Array,
    ) -> Array:
        result = [
            self.coefficients[i] * op._expect_kernel(logpsi, params, x)
            for i, op in enumerate(self.operators)
        ]
        return sum(result)

    @struct.property_cached(pytree_ignore=True)
    def _attrs(self) -> tuple[Hashable, ...]:
        return (  # type: ignore
            self.hilbert,
            self.operators,
            HashableArray(self.coefficients),
            self.dtype,
        )
