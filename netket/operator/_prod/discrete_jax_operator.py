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
from math import prod

import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.operator._discrete_operator_jax import DiscreteJaxOperator

from netket.operator._prod.base import ProductOperator


@register_pytree_node_class
class ProductDiscreteJaxOperator(ProductOperator, DiscreteJaxOperator):
    def __init__(self, *operators, coefficient=1.0, dtype=None):
        if not all(isinstance(op, DiscreteJaxOperator) for op in operators):
            raise TypeError(
                "Arguments to ProductDiscreteJaxOperator must all be "
                "subtypes of DiscreteJaxOperator. However the types are:\n\n"
                f"{list(type(op) for op in operators)}\n"
            )
        self._initialized = False
        super().__init__(
            operators, operators[0].hilbert, coefficient=coefficient, dtype=dtype
        )

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return prod(op.max_conn_size for op in self.operators)

    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_r = x.reshape(-1, 1, x.shape[-1])
        Ns, N = x_r.shape[0], x_r.shape[-1]
        samples = x
        mels = jnp.full((1, 1, 1), self.coefficient)
        for op in self.operators:
            op_ys, op_mels = op.get_conn_padded(samples)
            samples = op_ys.reshape(Ns, -1, N)
            mels = (mels * op_mels).reshape(Ns, -1, 1)

        ys = samples.reshape(*x.shape[:-1], -1, x.shape[-1])
        mels = jnp.concatenate(mels, axis=1).reshape(*ys.shape[:-1])
        return ys, mels

    def to_numba_operator(self) -> "ProductOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from netket.operator._prod.discrete_operator import ProductDiscreteOperator

        ops_numba = tuple(op.to_numba_operator() for op in self.operators)
        return ProductDiscreteOperator(
            *ops_numba, coefficients=self.coefficients, dtype=self.dtype
        )

    def tree_flatten(self):
        data = (self.operators, self.coefficient)
        metadata = {"dtype": self.dtype}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        ops, coefficient = data
        dtype = metadata["dtype"]

        # Create instance without calling __init__ to avoid ArgInfo validation issues
        obj = object.__new__(cls)
        obj._operators = tuple(ops)
        obj._coefficient = coefficient
        obj._dtype = dtype
        obj._initialized = False

        # Set the hilbert space from the first operator
        obj._hilbert = ops[0].hilbert

        return obj
