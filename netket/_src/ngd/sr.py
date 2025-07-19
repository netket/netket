from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P

from netket import jax as nkjax
from netket.utils.types import Array

from netket import config

from netket._src.ngd.kwargs import ensure_accepts_kwargs


@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
    ),
)
def _compute_sr_update(
    O_L,
    dv,
    *,
    diag_shift: float | Array,
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: float | Array | None = None,
    momentum: float | Array | None = None,
    old_updates: Array | None = None,
    params_structure,
):
    # We concretize the solver function to ensure it accepts the additional argument `dv`.
    # Typically solvers only accept the matrix and the right-hand side.
    solver_fn = ensure_accepts_kwargs(solver_fn, "dv")

    if (momentum is not None) or (old_updates is not None) or (proj_reg is not None):
        raise ValueError("Not implemented")

    # (np, #ns) x (#ns) -> (np) - where the sum over #ns is done automatically
    F = O_L.T @ dv

    # This does the contraction (np, #ns) x (#ns, np) -> (np, np).
    matrix = O_L.T @ O_L
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    shifted_matrix = jax.lax.add(
        matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
    )
    updates = solver_fn(shifted_matrix, F, dv=dv)

    # Some solvers return a tuple, some others do not.
    if isinstance(updates, tuple):
        updates, info = updates
        if info is None:
            info = {}
    else:
        info = {}

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if config.netket_experimental_sharding:
        updates = jax.lax.with_sharding_constraint(
            updates, NamedSharding(jax.sharding.get_abstract_mesh(), P())
        )

    return updates, old_updates, info
