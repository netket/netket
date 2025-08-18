# Copyright 2025 The NetKet Authors - All rights reserved.
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

from typing import Union
from collections.abc import Iterable

from netket.operator._abstract_operator import AbstractOperator
from netket.operator._sum.base import SumOperator


class SumGenericOperator(SumOperator, AbstractOperator):
    def __init__(
        self,
        *operators: AbstractOperator,
        coefficients: Union[float, Iterable[float]] = 1.0,
        dtype=None,
    ):
        if not all(isinstance(op, AbstractOperator) for op in operators):
            raise TypeError(
                "Arguments to SumOperator must all be subtypes of "
                "AbstractOperator. However the types are:\n\n"
                f"{list(type(op) for op in operators)}\n"
            )
        super().__init__(
            operators, operators[0].hilbert, coefficients=coefficients, dtype=dtype
        )
