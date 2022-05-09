# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Callable, Tuple
from functools import partial

import jax
from jax import numpy as jnp
import numpy as np

from netket.utils.types import PyTree


def segment_sumdiffexp(A_nz, B=None, *, row_lengths, H_nz=None):
    r"""
    computes
    $$
    \sum_{j=0}^{s[i]} H_{i,j}\exp[A_{i,j} - B_i]
    $$

    Args:
        vecA: a vector containing all non-zero entries in the matrix `A`.
        B: a vector containing all entries `B`
        sections: a keyword argument expressing the length of every row of the matrix A.
        H: an optional kewyrod argumnet that must be a vector with the same structure as `vecA`.
    """
    return _segment_sumdiffexp(A_nz, B, H_nz, row_lengths)


@partial(jax.custom_jvp, nondiff_argnums=(3,))
def _segment_sumdiffexp(A_nz, B, H_nz, row_lengths):

    if A_nz.ndim != 1:
        raise ValueError(
            "The first argument `A_nz` must be a vector of nonzero entries."
        )
    if H_nz is None:
        H_nz = 1
    elif H_nz.shape != A_nz.shape:
        raise ValueError(
            "The optional argument `H_nz` must have the same shape as `A_nz`."
        )

    N_rows = row_lengths.shape[0]
    n_nonzero = A_nz.size

    if B is None:
        B_ext = 1
    elif B.shape[0] == row_lengths.shape[0]:
        B_ext = jnp.repeat(B, row_lengths, total_repeat_length=n_nonzero)
    else:
        raise ValueError(
            "The second argument `B` must be a vector with the same length as `row_lengths`."
        )

    # compute the matrix elements in vector form
    values = H_nz * jnp.exp(A_nz - B_ext)

    # Construct the indices necessary to perform the segment_sum
    indices = jnp.repeat(jnp.arange(N_rows), row_lengths, total_repeat_length=n_nonzero)

    # sum contiguous blocks of `values` according to `indices`
    result = jax.ops.segment_sum(
        values, indices, num_segments=N_rows, indices_are_sorted=True
    )

    return result


@_segment_sumdiffexp.defjvp
def _segment_sumdiffexp_jvp(row_lengths, primals, tangents):
    print(f"extra {row_lengths}")
    A_nz, B, H_nz = primals
    A_nz_dot, B_dot, H_nz_dot = tangents

    primal_out = _segment_sumdiffexp(A_nz, B, H_nz, row_lengths)
    dR_dH_dH = _segment_sumdiffexp(A_nz, B, H_nz_dot, row_lengths)
    dR_dA_dA = _segment_sumdiffexp(A_nz, B, H_nz * A_nz_dot, row_lengths)
    dR_dB = -_segment_sumdiffexp(A_nz, B, H_nz, row_lengths)
    print(f"{primal_out = }")
    print(f"{dR_dH_dH = }")
    print(f"{dR_dA_dA =}")
    print(f"{dR_dB =}")
    print(f"{A_nz_dot =}")
    print(f"{B_dot =}")
    print(f"{H_nz_dot =}")
    tangent_out = dR_dH_dH + dR_dA_dA + dR_dB * B_dot
    tangent_out = dR_dB * B_dot

    print(f"{tangent_out = }")

    return primal_out, tangent_out
