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
one process per GPU. This module takes care of

- choosing the mesh describing that grid: NetKet's own mesh for the default
  ``(n_devices, 1)`` grid, which matches how NetKet shards QGT/NTK matrices, or
  a dedicated 2D mesh over the same devices otherwise;
- moving the linear problem onto that mesh, and the solution back;
- picking a valid tile size for the grid.
"""

import math
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from netket.utils.optional_deps import import_optional_dependency

# `jaxmg` 1.0 rewrote the whole interface on top of cuSOLVERMp. The 0.0.x series
# wrapped the deprecated cuSOLVERMg backend, driving many GPUs from a single
# process, and had a different API. 1.1.1 is required because it ships the
# cuSOLVERMp release fixing wrong results of `syevd` on large matrices, and
# because 1.1 added support for single-axis meshes, which we rely on.
JAXMG_MIN_VERSION = "1.1.1"

_JAXMG_VERSION_MSG = """NetKet uses the cuSOLVERMp interface introduced in `jaxmg` 1.0,
                    and requires at least 1.1.1, whose cuSOLVERMp fixes wrong
                    eigendecompositions of large matrices. Older releases
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


@dataclass(frozen=True)
class JaxmgGrid:
    """The cuSOLVERMp process grid on which `jaxmg` runs."""

    mesh: Mesh
    """Mesh whose devices are laid out as the process grid."""
    matrix_specs: P
    """Sharding of the matrix over `mesh`, as expected by `jaxmg`."""
    context_mesh: Mesh | None
    """Concrete counterpart of jax's context mesh, or None if there is none."""

    @property
    def shape(self) -> tuple[int, int]:
        """Shape ``(process_rows, process_cols)`` of the grid."""
        return tuple(
            1 if axis is None else self.mesh.shape[axis] for axis in self.matrix_specs
        )

    @property
    def is_context_mesh(self) -> bool:
        """Whether the grid lives on the context mesh, so nothing is moved."""
        return self.mesh == self.context_mesh


def _context_mesh() -> Mesh | None:
    """
    Concrete mesh matching jax's (abstract) context mesh, usually the one set
    by NetKet, or None if no mesh is set.

    :func:`jax.sharding.get_mesh` cannot be called inside of :func:`jax.jit`,
    so the mesh is rebuilt from :func:`jax.devices`, assuming the devices are
    laid out in that order, as in the mesh set by NetKet. Outside of
    :func:`jax.jit` that assumption is checked, as a mesh with its devices in
    another order would make jax fail with an obscure error under jit.
    """
    abstract_mesh = jax.sharding.get_abstract_mesh()
    if abstract_mesh.empty:
        return None
    devices = np.asarray(jax.devices())
    if abstract_mesh.size != devices.size:
        raise ValueError(
            f"The distributed solvers require the jax mesh to span all the "
            f"{devices.size} devices, but it spans {abstract_mesh.size}. Set a "
            "mesh over all of `jax.devices()` with `jax.sharding.set_mesh`."
        )
    mesh = Mesh(
        devices.reshape(abstract_mesh.axis_sizes),
        abstract_mesh.axis_names,
        axis_types=abstract_mesh.axis_types,
    )

    try:
        actual_mesh = jax.sharding.get_mesh()
    except ValueError:
        # Inside of jax.jit, where it cannot be checked.
        actual_mesh = None
    if actual_mesh is not None and not actual_mesh.empty and actual_mesh != mesh:
        raise ValueError(
            "The distributed solvers require the devices of the jax mesh to be "
            "laid out in the order of `jax.devices()`, as in the mesh set by "
            f"NetKet, but the mesh is {actual_mesh}. Build it with "
            "`jax.sharding.Mesh(np.asarray(jax.devices()).reshape(...), ...)`."
        )
    return mesh


def jaxmg_grid(process_grid: tuple[int, int] | None = None) -> JaxmgGrid:
    """
    Choose the mesh and matrix sharding describing the cuSOLVERMp process grid.

    Args:
        process_grid: shape ``(process_rows, process_cols)`` of the process
            grid. Defaults to ``(n_devices, 1)``, which matches the row-sharded
            layout NetKet uses for QGT/NTK matrices. That grid is laid on
            NetKet's own single-axis mesh, so the matrix is not moved at all.
            Other grids use a dedicated 2D mesh over the same devices.
    """
    n_devices = jax.device_count()
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

    context_mesh = _context_mesh()
    if (
        context_mesh is not None
        and len(context_mesh.axis_names) == 1
        and process_cols == 1
    ):
        # jaxmg (>= 1.1) accepts a single-axis mesh, which describes the
        # (n_devices, 1) grid.
        return JaxmgGrid(
            context_mesh, P(context_mesh.axis_names[0], None), context_mesh
        )

    # Keep the axis types of the context mesh, as those of the arrays moved
    # onto the grid mesh must be consistent with them.
    if context_mesh is None:
        axis_type = AxisType.Auto
    else:
        axis_type = context_mesh.axis_types[0]
    mesh = Mesh(
        np.asarray(jax.devices()).reshape(process_rows, process_cols),
        JAXMG_AXIS_NAMES,
        axis_types=(axis_type, axis_type),
    )
    return JaxmgGrid(mesh, P(*JAXMG_AXIS_NAMES), context_mesh)


def _is_explicit(mesh: Mesh) -> bool:
    return all(t == AxisType.Explicit for t in mesh.axis_types)


def _place(x: jax.Array, sharding: NamedSharding) -> jax.Array:
    """
    Constrain `x` to `sharding` on its own mesh, which requires
    :func:`jax.reshard` for Explicit mesh axes and
    :func:`jax.lax.with_sharding_constraint` for Auto ones.
    """
    if _is_explicit(sharding.mesh):
        return jax.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def to_grid(grid: JaxmgGrid, A: jax.Array, b: jax.Array):
    """
    Lay out the linear problem as `jaxmg` expects: `A` sharded over the process
    grid, and `b` on the grid mesh.
    """
    if grid.is_context_mesh:
        return _place(A, NamedSharding(grid.mesh, grid.matrix_specs)), b
    # Moving between meshes requires a device_put, both for Auto and Explicit
    # mesh axes.
    A = jax.device_put(A, NamedSharding(grid.mesh, grid.matrix_specs))
    b = jax.device_put(b, NamedSharding(grid.mesh, P()))
    return A, b


def grid_context(grid: JaxmgGrid) -> AbstractContextManager:
    """
    Context in which to operate on arrays living on the grid mesh.

    `jaxmg` enters its mesh by itself, but the arrays it returns live on the
    grid mesh, so post-processing them needs that mesh to be the context mesh
    too. :func:`jax.sharding.use_abstract_mesh` is used because, unlike
    :func:`jax.set_mesh`, it also works while tracing.
    """
    if grid.is_context_mesh:
        return nullcontext()
    return jax.sharding.use_abstract_mesh(grid.mesh.abstract_mesh)


def replicated_matmul(grid: JaxmgGrid, a: jax.Array, b: jax.Array) -> jax.Array:
    """
    Compute ``a @ b`` replicated on the grid mesh, where `a` or `b` may be
    sharded along the contracted axis, as the eigenvectors given by `jaxmg`.
    With Explicit mesh axes jax requires the output sharding of such a
    contraction to be given. Must be called inside of :func:`grid_context`.
    """
    sharding = NamedSharding(grid.mesh, P())
    if _is_explicit(grid.mesh):
        return jnp.matmul(a, b, out_sharding=sharding)
    return _place(a @ b, sharding)


def replicate(grid: JaxmgGrid, x: jax.Array) -> jax.Array:
    """
    Replicate a solution vector living on the grid mesh. Must be called inside
    of :func:`grid_context`.
    """
    return _place(x, NamedSharding(grid.mesh, P()))


def from_grid(grid: JaxmgGrid, x: jax.Array) -> jax.Array:
    """
    Bring a replicated solution back to the context mesh, where NetKet can use
    it. Must be called outside of :func:`grid_context`.
    """
    if grid.is_context_mesh or grid.context_mesh is None:
        return x
    return jax.device_put(x, NamedSharding(grid.context_mesh, P()))


def default_tile_size(n: int, grid: JaxmgGrid, *, max_tile_size: int) -> int:
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
        grid: the process grid.
        max_tile_size: upper bound on the tile size, limiting the redistribution
            scratch space (which grows linearly in the tile size).
    """
    process_rows, process_cols = grid.shape
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


def check_matrix_shardable(n: int, grid: JaxmgGrid, *, caller: str):
    """
    Check that an ``n x n`` matrix can be block-distributed over the grid.

    cuSOLVERMp needs every process to own the same number of rows and columns,
    so the matrix size must be divisible by both process-grid dimensions.
    `jaxmg` checks this too, but its error does not say how to fix it.

    Args:
        n: size of the (square) matrix.
        grid: the process grid.
        caller: name of the solver, used in the error message.
    """
    process_rows, process_cols = grid.shape
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
