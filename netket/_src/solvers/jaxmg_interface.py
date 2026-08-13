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

"""
Glue code between NetKet's distributed dense solvers and `jaxmg
<https://flatironinstitute.github.io/jaxmg/>`_, which exposes NVIDIA's
cuSOLVERMp multi-GPU dense linear algebra routines to jax.

cuSOLVERMp distributes a matrix over a **two-dimensional** grid of processes,
with one process per GPU, so this module takes care of

- building the 2D device mesh that `jaxmg` expects out of NetKet's devices;
- picking a valid tile size for that grid;
- entering the 2D mesh as jax's context mesh, which `jaxmg`'s internal
  :func:`jax.shard_map` calls require.
"""

import numpy as np

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from netket.utils.optional_deps import import_optional_dependency

# `jaxmg` 1.0 rewrote the whole interface on top of cuSOLVERMp. The 0.0.x series
# wrapped the deprecated cuSOLVERMg backend, driving many GPUs from a single
# process, and had a different API.
JAXMG_MIN_VERSION = "1.0.0"

_JAXMG_VERSION_MSG = """NetKet uses the cuSOLVERMp interface introduced in `jaxmg` 1.0.
                    Older releases (0.0.x) wrapped the now deprecated cuSOLVERMg
                    backend with a different API, and are only supported by
                    NetKet 3.22 and earlier."""

# Names of the two axes of the cuSOLVERMp process grid. cuSOLVERMp requires both
# matrix dimensions to be mapped onto a named mesh axis, so we cannot reuse
# NetKet's single-axis ('S') mesh and build a dedicated 2D mesh over the same
# devices instead.
JAXMG_AXIS_NAMES = ("jaxmg_rows", "jaxmg_cols")


def import_jaxmg(descr: str):
    """
    Import `jaxmg`, raising an informative error if it is missing or too old.

    Args:
        descr: description of the functionality requiring `jaxmg`.
    """
    return import_optional_dependency(
        "jaxmg",
        minimum_version=JAXMG_MIN_VERSION,
        descr=descr,
        extra_msg=_JAXMG_VERSION_MSG,
    )


def jaxmg_mesh(process_grid: tuple[int, int] | None = None) -> tuple[Mesh, P]:
    """
    Build the 2D device mesh and matrix :class:`~jax.sharding.PartitionSpec`
    describing the cuSOLVERMp process grid.

    Args:
        process_grid: shape ``(process_rows, process_cols)`` of the process
            grid. Defaults to ``(n_devices, 1)``, which matches the row-sharded
            layout NetKet uses for QGT/NTK matrices, and therefore requires no
            redistribution of the matrix.

    Returns:
        The tuple ``(mesh, matrix_specs)`` to be passed to `jaxmg`.
    """
    devices = np.asarray(jax.devices())
    n_devices = devices.size

    if process_grid is None:
        process_grid = (n_devices, 1)

    if len(process_grid) != 2:
        raise ValueError(
            "The `process_grid` must be a pair `(process_rows, process_cols)`, "
            f"but got {process_grid}."
        )
    process_rows, process_cols = (int(size) for size in process_grid)
    if process_rows * process_cols != n_devices:
        raise ValueError(
            f"The process grid ({process_rows}, {process_cols}) has "
            f"{process_rows * process_cols} slots, but there are {n_devices} "
            "devices. cuSOLVERMp uses one process per GPU, so the grid must "
            "contain exactly one slot per device."
        )

    mesh = Mesh(devices.reshape(process_rows, process_cols), JAXMG_AXIS_NAMES)
    return mesh, P(*JAXMG_AXIS_NAMES)


def jaxmg_mesh_context(mesh: Mesh):
    """
    Return a context manager entering `mesh` as jax's context mesh.

    `jaxmg` internally calls :func:`jax.shard_map` on `mesh`, which requires it
    to be the context mesh, while NetKet globally sets its own single-axis mesh.
    :func:`jax.sharding.use_abstract_mesh` is used because, unlike
    :func:`jax.set_mesh`, it also works while tracing (the solvers are usually
    called inside of :func:`jax.jit`).
    """
    return jax.sharding.use_abstract_mesh(mesh.abstract_mesh)


def replicate_solution(x, mesh: Mesh):
    """
    Replicate a solution vector living on the process grid `mesh`.

    `jaxmg` gives its results back sharded over the process grid. Replicating
    them, which is how NetKet stores parameter-space vectors anyway, also makes
    the result usable outside of the process-grid mesh.

    Must be called while `mesh` is the context mesh (see
    :func:`jaxmg_mesh_context`).
    """
    return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, P()))


def default_tile_size(n: int, mesh: Mesh, *, max_tile_size: int) -> int:
    """
    Default cuSOLVERMp square tile size for an ``n x n`` matrix.

    The largest tile that keeps every process-grid row and column owning at
    least one tile is ``n // max(process_rows, process_cols)``. On the default
    ``(n_devices, 1)`` grid this is the local number of rows, so tiles align
    with the jax shards and no padded copy of the matrix is allocated.

    Args:
        n: size of the (square) matrix.
        mesh: mesh describing the process grid.
        max_tile_size: upper bound on the tile size, limiting the redistribution
            scratch space (which grows linearly in the tile size).
    """
    local_rows, local_cols = (n // size for size in mesh.devices.shape)
    return max(1, min(local_rows, local_cols, max_tile_size))


def check_matrix_shardable(n: int, mesh: Mesh, *, caller: str):
    """
    Check that an ``n x n`` matrix can be block-distributed over the grid.

    cuSOLVERMp needs every process to own the same number of rows and columns,
    so the matrix size must be divisible by both process-grid dimensions.

    Args:
        n: size of the (square) matrix.
        mesh: mesh describing the process grid.
        caller: name of the solver, used in the error message.
    """
    process_rows, process_cols = mesh.devices.shape
    if n % process_rows != 0 or n % process_cols != 0:
        raise ValueError(
            f"`{caller}` cannot distribute a matrix of size {n} over a "
            f"({process_rows}, {process_cols}) cuSOLVERMp process grid: the "
            "matrix size must be divisible by both grid dimensions.\n\n"
            "This usually means that the number of samples (or parameters) is "
            "not a multiple of the number of GPUs. Either adjust it, or pass a "
            "compatible `process_grid=(process_rows, process_cols)` to the "
            "solver."
        )
