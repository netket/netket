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

cuSOLVERMp distributes a matrix over a two-dimensional grid of processes, with
one process per GPU. `jaxmg` reads that grid off the sharding of the matrix (or
off the context mesh inside :func:`jax.jit`), so the default ``(n_devices, 1)``
grid, which matches how NetKet shards QGT/NTK matrices, is simply NetKet's mesh.
This module takes care of

- moving the linear problem onto a dedicated 2D mesh for other grids, and the
  solution back;
- picking a valid tile size for the grid.
"""

import math
from collections.abc import Callable

import jax
from jax.sharding import AxisType, NamedSharding, PartitionSpec as P

from netket.utils.optional_deps import import_optional_dependency

# 1.4 is the first release inferring the mesh and the sharding of the matrix by
# itself, also inside of `jax.jit`, and 1.4.1 fills the outputs with NaNs when
# the native solver fails, which `nan_fallback` relies on.
JAXMG_MIN_VERSION = "1.4.1"

_JAXMG_VERSION_MSG = """Install jaxmg with the CUDA version of your jax, which also
                    installs NVIDIA's cuSOLVERMp library:
                    `pip install 'jaxmg[cuda12]'` or `pip install 'jaxmg[cuda13]'`.

                    NetKet requires at least `jaxmg` 1.4.1, which infers the mesh
                    and the sharding of the matrix inside of `jax.jit`, and returns
                    NaNs when the native solver fails. Older releases
                    (0.0.x) wrapped the now deprecated cuSOLVERMg backend with a
                    different API, and are only supported by NetKet 3.22 and
                    earlier."""

# Names of the axes of the dedicated mesh used for genuinely 2D process grids.
JAXMG_AXIS_NAMES = ("jaxmg_rows", "jaxmg_cols")

# Tiles smaller than this are slow enough that padding the matrix to a larger
# tile is preferable.
_MIN_TILE_SIZE = 64


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


def process_grid_shape(process_grid: tuple[int, int] | None) -> tuple[int, int]:
    """
    Validate the shape ``(process_rows, process_cols)`` of the cuSOLVERMp process
    grid, defaulting to ``(n_devices, 1)``.
    """
    n_devices = jax.device_count()
    if process_grid is None:
        return (n_devices, 1)

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
    return (process_rows, process_cols)


def on_process_grid(
    fn: Callable,
    process_grid: tuple[int, int] | None,
    A: jax.Array,
    b: jax.Array,
) -> jax.Array:
    """
    Compute ``fn(A, b)``, which calls `jaxmg`, on the given process grid, and
    return its result on the context mesh.

    `jaxmg` lays `A` over the process grid described by its sharding, or by the
    context mesh inside of :func:`jax.jit`. By default, `A` is handed to `jaxmg`
    as it is on NetKet's single-axis mesh, which describes the
    ``(n_devices, 1)`` grid (or ``(1, n_devices)`` if `A` is sharded by
    columns, which needs no redistribution either). An explicit
    ``(n_devices, 1)`` grid also lives on that mesh, with the rows of `A`
    sharded over it. Other grids live on a dedicated 2D mesh, with the axis
    types of the context mesh: `A` and `b` are moved onto it, and ``fn`` runs
    with it as the context mesh so that it can post-process the arrays given
    by `jaxmg`. jax requires all the arrays of a computation to have their
    devices in the same order, and that mesh lays out :func:`jax.devices` in
    order, so the context mesh must do so too, as NetKet's mesh does.
    """
    grid_shape = process_grid_shape(process_grid)
    context_mesh = jax.sharding.get_abstract_mesh()
    if len(context_mesh.axis_names) == 1 and grid_shape[1] == 1:
        if process_grid is not None:
            row_specs = P(context_mesh.axis_names[0], None)
            if context_mesh.axis_types[0] == AxisType.Explicit:
                A = jax.reshard(A, row_specs)
            else:
                A = jax.lax.with_sharding_constraint(A, row_specs)
        return fn(A, b)

    axis_type = AxisType.Auto if context_mesh.empty else context_mesh.axis_types[0]
    mesh = jax.make_mesh(
        grid_shape, JAXMG_AXIS_NAMES, axis_types=(axis_type, axis_type)
    )
    A = jax.device_put(A, NamedSharding(mesh, P(*JAXMG_AXIS_NAMES)))
    b = jax.device_put(b, NamedSharding(mesh, P()))
    with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
        x = fn(A, b)
        # With Auto mesh axes, moving `x` back to the context mesh only gives
        # it a sharding expressible there if it is replicated first. With
        # Explicit ones, it is already replicated.
        if axis_type == AxisType.Auto:
            x = jax.lax.with_sharding_constraint(x, P())
    if context_mesh.empty:
        return x
    return jax.device_put(x, P())


def default_tile_size(
    n: int, grid_shape: tuple[int, int], *, max_tile_size: int
) -> int:
    """
    Default cuSOLVERMp square tile size for an ``n x n`` matrix.

    `jaxmg` pads every local shard of the matrix to a multiple of the tile
    size, allocating a padded copy of it. To avoid that, this returns the
    largest tile dividing both local dimensions, which also guarantees that
    every process row and column owns at least one tile. On the default
    ``(n_devices, 1)`` grid, this is the local number of rows if it does not
    exceed `max_tile_size`.

    If the only such tiles are smaller than ``64``, which would be slow, the
    matrix is padded instead, choosing the tile that minimises the padding
    among those splitting the local shard in as few tiles as allowed by
    `max_tile_size`.

    Args:
        n: size of the (square) matrix.
        grid_shape: shape ``(process_rows, process_cols)`` of the process grid.
        max_tile_size: upper bound on the tile size, limiting the redistribution
            scratch space (which grows linearly in the tile size).
    """
    process_rows, process_cols = grid_shape
    local_size = math.gcd(n // process_rows, n // process_cols)
    largest = max(1, min(local_size, max_tile_size))
    divisor = max(
        d
        for i in range(1, math.isqrt(local_size) + 1)
        if local_size % i == 0
        for d in (i, local_size // i)
        if d <= largest
    )
    if divisor < min(_MIN_TILE_SIZE, largest):
        n_tiles = -(-local_size // largest)
        return -(-local_size // n_tiles)
    return divisor


def check_matrix_shardable(n: int, grid_shape: tuple[int, int], *, caller: str):
    """
    Check that an ``n x n`` matrix can be block-distributed over the grid.

    cuSOLVERMp needs every process to own the same number of rows and columns,
    so the matrix size must be divisible by both process-grid dimensions.
    `jaxmg` checks this too, but its error does not say how to fix it.

    Args:
        n: size of the (square) matrix.
        grid_shape: shape ``(process_rows, process_cols)`` of the process grid.
        caller: name of the solver, used in the error message.
    """
    process_rows, process_cols = grid_shape
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
