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

from functools import partial

import jax

from netket.optimizer.linear_operator import DenseOperator
from netket.utils.api_utils import partial_from_kwargs

from netket._src.vqs.fermion_mf.state import DeterminantVariationalState


@partial_from_kwargs
def QGTByWick(
    vstate: DeterminantVariationalState,
    *,
    diag_shift: float = 0.0,
    **kwargs,
) -> DenseOperator:
    """
    Computes the Quantum Geometric Tensor (QGT) for a fermionic mean-field state
    analytically using Wick's theorem.

    This function follows the same API pattern as other QGT constructors like
    :func:`~netket.optimizer.qgt.QGTJacobianDense`.

    Args:
        vstate: The DeterminantVariationalState for which to compute the QGT.
        diag_shift: Diagonal shift to add to the QGT matrix for regularization.
            This is equivalent to adding a small positive value to the diagonal
            to ensure the matrix is positive definite (default: 0.0).
        **kwargs: Additional keyword arguments (currently unused, for API compatibility).

    Returns:
        A DenseOperator wrapping the dense QGT matrix.

    """
    if kwargs.pop("diag_scale", None) is not None:
        raise NotImplementedError(
            "\n`diag_scale` argument is not yet supported by QGTByWick.\n"
        )

    qgt_matrix = _jit_apply(
        vstate._model,
        vstate.variables,
        method=vstate._model.qgt_by_wick,
    )

    return DenseOperator(matrix=qgt_matrix, diag_shift=diag_shift)


@partial(jax.jit, static_argnames=("model", "method"))
def _jit_apply(model, *args, method):
    return model.apply(*args, method=method)
