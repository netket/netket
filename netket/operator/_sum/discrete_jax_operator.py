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

import numpy as np
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.operator._discrete_operator_jax import DiscreteJaxOperator

from netket.operator._sum.base import SumOperator


@register_pytree_node_class
class SumDiscreteJaxOperator(SumOperator, DiscreteJaxOperator):
    def __init__(self, *operators, coefficients=1.0, dtype=None):
        if not all(isinstance(op, DiscreteJaxOperator) for op in operators):
            raise TypeError(
                "Arguments to SumDiscreteJaxOperator must all be "
                "subtypes of DiscreteJaxOperator. However the types are:\n\n"
                f"{list(type(op) for op in operators)}\n"
            )
        self._initialized = False
        super().__init__(
            operators, operators[0].hilbert, coefficients=coefficients, dtype=dtype
        )

    def _setup(self, force: bool = False):
        self._initialized = True
        for op in self.operators:
            if hasattr(op, "_setup"):
                op._setup(force=force)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return sum(op.max_conn_size for op in self.operators)

    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_r = x.reshape(-1, x.shape[-1])
        ys = []
        mels = []
        for op, c in zip(self.operators, self.coefficients):
            op_ys, op_mels = op.get_conn_padded(x_r)
            ys.append(op_ys)
            mels.append(c * op_mels)

        ys = jnp.concatenate(ys, axis=1).reshape(*x.shape[:-1], -1, x.shape[-1])
        mels = jnp.concatenate(mels, axis=1).reshape(*ys.shape[:-1])
        return ys, mels

    def to_numba_operator(self) -> "SumOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from .discrete import SumDiscreteOperator

        ops_numba = tuple(op.to_numba_operator() for op in self.operators)
        return SumDiscreteOperator(
            *ops_numba, coefficients=self.coefficients, dtype=self.dtype
        )

    def tree_flatten(self):
        data = (self.operators, self.coefficients)
        metadata = {"dtype": self.dtype}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        ops, coefficients = data
        dtype = metadata["dtype"]

        return cls(*ops, coefficients=coefficients, dtype=dtype)
