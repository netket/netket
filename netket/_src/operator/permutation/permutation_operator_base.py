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

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import AbstractHilbert
from netket.operator import DiscreteJaxOperator
from netket.symmetry.group import Permutation, Identity


@register_pytree_node_class
class PermutationOperatorBase(DiscreteJaxOperator):
    """
    Abstract permutation operator for either spin of fermion space.
    For mathematical details on the definition of a permutation operator
    and its justification, we refer to [SOMEWHERE].

    Args:
        hilbert: The Hilbert space.
        permutation: The permutation represented by the operator.
    """

    def __init__(self, hilbert: AbstractHilbert, permutation: Permutation):
        """
        Constructs a representation of the given permutation acting on kets
        of the given hilbert space.

        Args:
            hilbert: The Hilbert space.
            permutation: The permutation represented by the operator.
        """
        if isinstance(permutation, Identity):
            permutation = Permutation(
                permutation_array=jnp.arange(hilbert.size), name="Identity"
            )

        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a Permutation object.")
        if not hilbert.size == permutation.permutation_array.size:
            raise ValueError(
                "Permutation size does not correspond to Hilbert space size."
            )

        super().__init__(hilbert)
        self.permutation = permutation

    def tree_flatten(self):
        struct_data = {"hilbert": self.hilbert, "permutation": self.permutation}
        return (), struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        return cls(**struct_data)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return jnp.float32

    def __hash__(self):
        return hash((self.hilbert, self.permutation))

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.hilbert == other.hilbert and self.permutation == other.permutation
            )

    def __matmul__(self, other):
        if type(self) is type(other) and self.hilbert == other.hilbert:
            return type(self)(self.hilbert, self.permutation @ other.permutation)
        else:
            return super().__matmul__(other)

    def trace(self) -> float:
        """
        Computes the trace of the operator on the given Hilbert space.

        This could also raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            f"trace method not implemented for class `{type(self)}`"
        )
