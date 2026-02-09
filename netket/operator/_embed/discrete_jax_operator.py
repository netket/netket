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

from functools import reduce

import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import BCOO, BCSR, JAXSparse
from jax.tree_util import register_pytree_node_class
from scipy.sparse import csr_matrix as _csr_matrix

from netket.hilbert import TensorHilbert

from .._discrete_operator_jax import DiscreteJaxOperator

from .base import EmbedOperator


@register_pytree_node_class
class EmbedDiscreteJaxOperator(EmbedOperator, DiscreteJaxOperator):
    def __init__(
        self,
        hilbert: TensorHilbert,
        operator: DiscreteJaxOperator,
        subspace: int,
    ):

        if not isinstance(operator, DiscreteJaxOperator):
            raise TypeError(
                "Arguments to EmbedDiscreteJaxOperator must be "
                "subtypes of DiscreteJaxOperator. However the type is:\n\n"
                f"{type(operator)}\n"
            )
        super().__init__(hilbert, operator, subspace)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        return self.operator.max_conn_size

    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_sub = x[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ]
        x_conn_sub, mels = self.operator.get_conn_padded(x_sub)
        x_conn = x.reshape(*x.shape[:-1], 1, x.shape[-1])
        x_conn = x_conn.repeat(x_conn_sub.shape[-2], axis=-2)
        x_conn = x_conn.at[
            ...,
            self.hilbert._cum_indices[self.subspace] : self.hilbert._cum_indices[
                self.subspace + 1
            ],
        ].set(x_conn_sub)
        return x_conn, mels

    def to_sparse(self, jax_: bool = False) -> JAXSparse | _csr_matrix:
        # Fast specialized implementation based on direct kronecker products
        if not jax_:
            return super().to_sparse()
        else:
            op_sparse = self.operator.to_sparse(jax_=True)

            result_bcoo = reduce(
                _sparse_kron_bcoo,
                (
                    (
                        op_sparse
                        if i == self.subspace
                        else _jax_sparse_identity(sub.n_states, op_sparse.dtype)
                    )
                    for i, sub in enumerate(self.hilbert.subspaces)
                ),
            )

            # Clean up and convert to BCSR
            result_bcoo = result_bcoo.sum_duplicates().update_layout(n_batch=0)
            return BCSR.from_bcoo(result_bcoo)

    def to_numba_operator(self) -> "EmbedDiscreteOperator":  # noqa: F821
        """
        Returns the standard (numba) version of this operator, which is an
        instance of {class}`nk.operator.Ising`.
        """
        from .discrete_operator import EmbedDiscreteOperator

        return EmbedDiscreteOperator(
            self.hilbert,
            self.operator.to_numba_operator(),
            self.subspace,
        )

    def tree_flatten(self):
        data = (self.operator,)
        metadata = {"hilbert": self.hilbert, "subspace": self.subspace}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (operator,) = data
        hilbert = metadata["hilbert"]
        subspace = metadata["subspace"]

        return cls(hilbert, operator, subspace)


# jax sparse matmul utilities
def _jax_sparse_identity(n, dtype):
    """Create sparse identity matrix in BCOO format."""
    indices = jnp.stack([jnp.arange(n), jnp.arange(n)], axis=1)
    data = jnp.ones(n, dtype=dtype)
    return BCOO((data, indices), shape=(n, n))


def _sparse_kron_bcoo(A: BCOO | BCSR, B: BCOO | BCSR) -> BCOO:
    """Compute sparse Kronecker product of two sparse matrices without going through dense.

    For sparse A (m×n) and B (p×q), kron(A,B) is (mp×nq) where each A[i,j]
    is replaced by the scaled matrix A[i,j] * B.

    Args:
        A: First sparse matrix (BCOO or BCSR)
        B: Second sparse matrix (BCOO or BCSR)

    Returns:
        Kronecker product as BCOO sparse matrix
    """
    # Convert to BCOO if needed (BCOO has simpler indexing)
    if not isinstance(A, BCOO):
        A = A.to_bcoo()
    if not isinstance(B, BCOO):
        B = B.to_bcoo()

    # Extract indices and data from BCOO format
    A_indices = A.indices  # shape: (nnz_A, 2)
    A_data = A.data  # shape: (nnz_A,)
    B_indices = B.indices  # shape: (nnz_B, 2)
    B_data = B.data  # shape: (nnz_B,)

    # Compute all combinations via broadcasting
    # A_indices[:, None, :] has shape (nnz_A, 1, 2)
    # B_indices[None, :, :] has shape (1, nnz_B, 2)
    A_i = A_indices[:, None, 0]  # (nnz_A, 1) - row indices of A
    A_j = A_indices[:, None, 1]  # (nnz_A, 1) - col indices of A
    B_i = B_indices[None, :, 0]  # (1, nnz_B) - row indices of B
    B_j = B_indices[None, :, 1]  # (1, nnz_B) - col indices of B

    # Compute Kronecker product indices
    new_i = A_i * B.shape[0] + B_i  # (nnz_A, nnz_B)
    new_j = A_j * B.shape[1] + B_j  # (nnz_A, nnz_B)
    new_indices = jnp.stack([new_i.ravel(), new_j.ravel()], axis=1)

    # Compute Kronecker product data (outer product)
    new_data = (A_data[:, None] * B_data[None, :]).ravel()

    # New shape
    new_shape = (A.shape[0] * B.shape[0], A.shape[1] * B.shape[1])

    return BCOO((new_data, new_indices), shape=new_shape)
