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

from typing import Tuple, Iterable

from jax import numpy as jnp

from netket.utils.types import DType
from netket.vqs import VariationalState

from netket.operator import (
    AbstractOperator,
)


class OperatorWithPenalty(AbstractOperator):
    """...."""

    def __init__(
        self,
        operator: AbstractOperator,
        states: Iterable[VariationalState],
        shifts: Iterable[float],
    ):
        super().__init__(operator.hilbert)

        if len(states) != len(shifts):
            raise ValueError("Number of states and shifts must be the same.")

        self._bare_operator = operator

        self._states = tuple(states)
        self._shifts = jnp.asarray(shifts)

    @property
    def states(self) -> Tuple[VariationalState]:
        return self._states

    @property
    def shifts(self) -> Tuple[VariationalState]:
        return self._shifts

    @property
    def operator(self) -> AbstractOperator:
        return self._bare_operator

    @property
    def is_hermitian(self) -> bool:
        return self.operator.is_hermitian

    @property
    def dtype(self) -> DType:
        return self.operator.dtype
