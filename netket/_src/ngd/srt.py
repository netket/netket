from typing import Callable, Optional
from functools import partial

import jax
import jax.numpy as jnp
from jax.sharding import PositionalSharding

from netket import jax as nkjax
from netket.utils import mpi
from netket.utils.types import Union, Array

from netket._src import distributed as distributed


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
    diag_shift: Union[float, Array],
    solver_fn: Callable[[Array, Array], Array],
    mode: str,
    proj_reg: Optional[Union[float, Array]] = None,
    momentum: Optional[Union[float, Array]] = None,
    old_updates: Optional[Array] = None,
    params_structure,
):
    if momentum is not None:
        dv -= momentum * (O_L @ old_updates)

    # Equivalent to MPI.alltoall, shards the data across axis 1
    # (#ns, np) -> (ns, #np)
    O_LT, token = distributed.reshard(
        O_L, sharded_axis=0, out_sharded_axis=1, pad=True, pad_value=0.0
    )
    dv, token = distributed.allgather(dv, token=token)

    # This does the contraction (ns, #np) x (#np, ns) -> (ns, ns).
    # When using sharding the sum over #ns is done automatically.
    # When using MPI we need to do it manually with an allreduce_sum.
    matrix, token = mpi.mpi_reduce_sum_jax(O_LT @ O_LT.T, root=0, token=token)
    matrix_side = matrix.shape[-1]  # * it can be ns or 2*ns, depending on mode

    if mpi.rank == 0:
        shifted_matrix = jax.lax.add(
            matrix, diag_shift * jnp.eye(matrix_side, dtype=matrix.dtype)
        )

        if proj_reg is not None:
            shifted_matrix = jax.lax.add(
                shifted_matrix, jnp.full_like(shifted_matrix, proj_reg / matrix_side)
            )

        aus_vector = solver_fn(shifted_matrix, dv)

        # Some solvers return a tuple, some others do not.
        if isinstance(aus_vector, tuple):
            aus_vector, info = aus_vector
        else:
            info = {}

        aus_vector = aus_vector.reshape(mpi.n_nodes, -1)
        aus_vector, token = mpi.mpi_scatter_jax(aus_vector, root=0, token=token)
    else:
        aus_vector = jnp.zeros((int(matrix_side / mpi.n_nodes),), dtype=jnp.float64)
        aus_vector, token = mpi.mpi_scatter_jax(aus_vector, root=0, token=token)
        info = None

    if info is None:
        info = {}

    # (np, #ns) x (#ns) -> (np).
    # The sum over #ns is done automatically in sharding.
    # Under MPI we need to do it manually with an allreduce_sum.
    updates, token = mpi.mpi_allreduce_sum_jax(O_L.T @ aus_vector, token=token)
    if momentum is not None:
        updates += momentum * old_updates
        old_updates = updates

    # If complex mode and we have complex parameters, we need
    # To repack the real coefficients in order to get complex updates
    if mode == "complex" and nkjax.tree_leaf_iscomplex(params_structure):
        num_p = updates.shape[-1] // 2
        updates = updates[:num_p] + 1j * updates[num_p:]

    if distributed.mode() == "sharding":
        out_shardings = (
            PositionalSharding(jax.devices()).replicate().reshape((1,) * updates.ndim)
        ).replicate()
        updates = jax.lax.with_sharding_constraint(updates, out_shardings)

    return updates, old_updates, info
