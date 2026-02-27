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

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from netket.utils.partial import HashablePartial


def _nan_fallback_solver(primary_solver, fallback_solver, A, b, *, x0=None):
    b_flat, _ = ravel_pytree(b)

    x_primary, info_primary = primary_solver(A, b, x0=x0)
    x_primary_flat, unravel = ravel_pytree(x_primary)

    input_bad = jnp.any(jnp.isnan(b_flat))
    if isinstance(A, jax.Array):
        input_bad = input_bad | jnp.any(jnp.isnan(A))
    output_bad = jnp.any(jnp.isnan(x_primary_flat)) | jnp.any(jnp.isinf(x_primary_flat))
    solver_fallback = input_bad | output_bad

    x_flat = jax.lax.cond(
        solver_fallback,
        lambda: ravel_pytree(fallback_solver(A, b, x0=x0)[0])[0],
        lambda: x_primary_flat,
    )

    if info_primary is None:
        combined_info = {"solver_fallback": solver_fallback}
    else:
        combined_info = {**info_primary, "solver_fallback": solver_fallback}

    return unravel(x_flat), combined_info


def nan_fallback(primary_solver, fallback_solver):
    r"""
    Creates a solver that transparently falls back to a more robust solver when
    the primary produces NaN or Inf.

    The primary solver always runs. If its output contains NaN or Inf (or the
    right-hand side ``b`` contains NaN), the fallback solver is invoked instead
    via :func:`jax.lax.cond`, so it only executes at runtime when needed.

    The returned info dict always contains a ``solver_fallback`` key indicating
    whether the fallback was activated. Any additional entries from the primary
    solver's info are also included. Info from the fallback solver is not
    included, to avoid running it unconditionally.

    The returned solver supports equality and hashing, so that two calls with
    the same solvers produce equal objects and do not trigger JAX recompilation.

    Args:
        primary_solver: The preferred (usually faster) solver.
        fallback_solver: The robust solver to use when the primary fails.

    Returns:
        A new solver function with the same signature as the inputs.

    Example:
        Create a Cholesky solver that falls back to the pseudo-inverse when
        numerical issues arise:

        >>> solver = nan_fallback(cholesky, pinv_smooth(rtol=1e-6))
        >>> x, info = solver(A, b)
        >>> if info["solver_fallback"]:
        ...     print("Cholesky failed, used pinv_smooth instead")
    """
    return HashablePartial(_nan_fallback_solver, primary_solver, fallback_solver)
