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

from jax.tree_util import register_pytree_node_class

from netket.hilbert import AbstractHilbert
from netket.operator import DiscreteJaxOperator
from netket.symmetry.group import Permutation, Identity
from netket.jax import canonicalize_dtypes
from netket.utils.types import DType


@register_pytree_node_class
class PermutationOperatorBase(DiscreteJaxOperator):
    """
    Base class for permutation operators acting on an
    Hilbert space. Used to express lattice symmetries and symmetrize
    variational states with respect to these symmetries.

    Subclasses implement this on Spin, Fermionic or other hilbert spaces.
    Look at :class:`netket.operator.permutation.PermutationOperator` and
    :class:`netket.operator.permutation.PermutationOperatorFermion` for
    examples.

    For mathematical details on the definition of a permutation operator
    and its justification, we refer to :doc:`/advanced/symmetry`.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        permutation: Permutation,
        *,
        dtype: DType | None = float,
    ):
        """
        Construct the permutation operator.

        Args:
            hilbert: The Hilbert space.
            permutation: The permutation that defines the operator as
                specified in :doc:`/advanced/symmetry`.
        """
        if isinstance(permutation, Identity):
            permutation = Permutation(
                permutation_array=np.arange(hilbert.size),
                name="Identity",
                validate=False,
            )

        if not isinstance(permutation, Permutation):
            raise TypeError("permutation must be a Permutation object.")
        if not hilbert.size == permutation.permutation_array.size:
            raise ValueError(
                "Permutation size does not correspond to Hilbert space size."
            )

        dtype = canonicalize_dtypes(dtype=dtype)

        super().__init__(hilbert)
        self.permutation = permutation
        self._dtype = dtype

    def __repr__(self):
        if self.permutation._name is not None:
            perm_repr = f"{self.permutation._name}:{self.permutation.permutation_array}"
        else:
            perm_repr = f"{self.permutation.permutation_array}"
        return f"{type(self).__name__}({self.hilbert}, {perm_repr}, dtype={self.dtype})"

    def tree_flatten(self):
        struct_data = {
            "hilbert": self.hilbert,
            "permutation": self.permutation,
            "dtype": self.dtype,
        }
        return (), struct_data

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        return cls(**struct_data)

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return self._dtype

    def __hash__(self):
        return hash((self.hilbert, self.permutation))

    def __eq__(self, other):
        if type(self) is type(other):
            return (
                self.hilbert == other.hilbert and self.permutation == other.permutation
            )
        return False

    def __matmul__(self, other):
        if type(self) is type(other) and self.hilbert == other.hilbert:
            return type(self)(self.hilbert, self.permutation @ other.permutation)
        else:
            return super().__matmul__(other)

    def trace(self) -> complex:
        """
        Compute the trace of the operator on the given Hilbert space.

        This could also raise a `NotImplementedError`.
        """
        raise NotImplementedError(
            f"Trace method not implemented for class `{type(self)}`"
        )
