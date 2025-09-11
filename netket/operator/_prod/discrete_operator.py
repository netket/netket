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

import numpy as np

from netket.operator._discrete_operator import DiscreteOperator

from netket.operator._prod.base import ProductOperator


class ProductDiscreteOperator(ProductOperator, DiscreteOperator):
    def __init__(self, *operators, coefficient=1.0, dtype=None):
        if not all(isinstance(op, DiscreteOperator) for op in operators):
            raise TypeError(
                "Arguments to ProductDiscreteOperator must all be "
                "subtypes of DiscreteOperator. However the types are:\n\n"
                f"{list(type(op) for op in operators)}\n"
            )
        self._initialized = False
        super().__init__(
            operators, operators[0].hilbert, coefficient=coefficient, dtype=dtype
        )

    def _setup(self, force: bool = False):
        if not self._initialized:
            self._initialized = True

    def get_conn_flattened(
        self,
        x: np.ndarray,
        sections: np.ndarray,
        pad: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def to_jax_operator(self) -> "ProductDiscreteJaxOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from netket.operator._prod.discrete_jax_operator import (
            ProductDiscreteJaxOperator,
        )

        ops_jax = tuple(op.to_jax_operator() for op in self.operators)
        return ProductDiscreteJaxOperator(
            *ops_jax, coefficient=self.coefficient, dtype=self.dtype
        )
