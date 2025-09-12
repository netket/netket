from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec as P, get_abstract_mesh, reshard

from netket import jax as nkjax
from netket import config
from netket.utils import timing
from netket.utils.types import Array


@timing.timed
@partial(
    jax.jit,
    static_argnames=(
        "solver_fn",
        "mode",
    ),
)
def _compute_srt_update(
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
    if momentum is not None:
        dv -= momentum * (O_L @ old_updates)

    # Here we reshard the jacobian from being sharded along the sample axis to being
    # sharded along the parameter axis, which we pad
    # (#ns, np) -> (ns, #np)
    # In theory jax could figure this out, but maybe he does not, so we do it by hand.
    O_LT = O_L
    if config.netket_experimental_sharding:
        # nkjax.sharding.pad_axis_for_sharding(
        #    O_LT, axis=1, axis_name="S", padding_value=0.0
        # )
        if get_abstract_mesh().are_all_axes_explicit:
            # O_LT = reshard(O_LT, P(None, "S"))
            dv = reshard(dv, P())
        # O_LT = jax.lax.with_sharding_constraint(
        #     O_LT,
        #     NamedSharding(get_abstract_mesh(), P(None, "S")),
        # )
        dv = jax.lax.with_sharding_constraint(
            dv, NamedSharding(get_abstract_mesh(), P())
        )

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix = jnp.matmul(O_LT, O_LT.T, out_sharding=P())
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    shifted_matrix = jax.lax.add(
        matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
    )
    # replicate

    if proj_reg is not None:
        shifted_matrix = jax.lax.add(
            shifted_matrix, jnp.full_like(shifted_matrix, proj_reg / matrix_side)
        )

    aus_vector = solver_fn(shifted_matrix, dv)

    # Some solvers return a tuple, some others do not.
    if isinstance(aus_vector, tuple):
        aus_vector, info = aus_vector
        if info is None:
            info = {}
    else:
        info = {}

    # (np, #ns) x (#ns) -> (np).
    if get_abstract_mesh().are_all_axes_explicit:
        aus_vector = reshard(aus_vector, P("S"))
    updates = jnp.matmul(O_L.T, aus_vector, out_sharding=jax.P())
    if momentum is not None:
        updates += momentum * old_updates
        old_updates = updates

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if config.netket_experimental_sharding:
        updates = jax.lax.with_sharding_constraint(
            updates, NamedSharding(get_abstract_mesh(), P())
        )

    return updates, old_updates, info
