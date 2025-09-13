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

from netket.hilbert import HomogeneousHilbert
from netket.hilbert.constraint import SumConstraint

from netket._src.operator.permutation.permutation_operator_base import (
    PermutationOperatorBase,
)
from netket._src.operator.permutation.trace_utils import count_n_uplets


@register_pytree_node_class
class PermutationOperator(PermutationOperatorBase):
    """
    Permutation operator on a spin or boson space. Used for
    the symmetry-representation machinery.

    For mathematical details on the definition of a permutation operator
    and its justification, we refer to :doc:`/advanced/symmetry`.

    For the fermionic counterpart look at
    :class:`netket.operator.permutation.PermutationOperatorFermion`.
    """

    def get_conn_padded(self, x):
        r"""Finds the connected elements of the Operator.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        .. warning::

            Unlike most other operators defined in NetKet, a permutation operator
            is not Hermitian, and we thus have to be careful about the definition of
            connected elements. NetKet defines connected elements of :math:`x` as the
            configurations :math:`x'` such that :math:`\langle x | P_\sigma | x' \rangle` .
            Therefore, the connected elements are the configurations found in the
            image of :math:`x` by :math:`P_{\sigma^{-1}}` , and not :math:`P_\sigma` .

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.

        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """
        x = jnp.asarray(x)
        # Check that the parameters of get are useful
        x_conn = x.at[..., None, self.permutation.permutation_array].get(
            unique_indices=True, mode="promise_in_bounds"
        )

        # we want to do
        # mels = jnp.ones((*x.shape[:-1], 1), dtype=self.dtype)
        # but to preserve sharding we must do
        mels = x.at[..., :1].get(unique_indices=True, mode="promise_in_bounds")
        mels = 1 + (mels * 0).astype(self.dtype)

        return x_conn, mels

    def trace(self) -> int:
        cycle_decomposition = self.permutation.cycle_decomposition()

        if isinstance(self.hilbert, HomogeneousHilbert):

            if not self.hilbert.constrained:
                return self.hilbert.local_size ** len(cycle_decomposition)

            elif isinstance(self.hilbert.constraint, SumConstraint):
                cycle_lengths = [len(cycle) for cycle in cycle_decomposition]
                shifted_constraint = (
                    self.hilbert.constraint.sum_value
                    + self.hilbert.size * (self.hilbert.local_size - 1)
                )
                return count_n_uplets(
                    cycle_lengths, shifted_constraint, self.hilbert.local_size
                )

            else:
                raise NotImplementedError(
                    f"Unimplemented trace for Hilbert constraint {type(self.hilbert.constraint)}"
                )

        else:
            raise NotImplementedError(
                f"Unimplemented trace for Hilbert space {type(self.hilbert).__name__}"
            )
