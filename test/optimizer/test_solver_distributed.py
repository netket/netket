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

"""Tests for distributed solvers (cholesky_distributed, pinv_smooth_distributed)."""

import re
import sys
from functools import partial
from types import ModuleType

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

import netket as nk
from netket._src.solvers.jaxmg_interface import default_tile_size

from test import common  # noqa: F401

# jaxmg's cuSOLVERMp kernels only run on CUDA GPUs, and only with one process
# per GPU (launch pytest with `djaxrun`, or `srun` with one task per GPU), so
# these tests are skipped otherwise.
requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu" or jax.local_device_count() != 1,
    reason="jaxmg distributed solvers require a GPU backend with one process per GPU",
)

N_DEVICES = jax.device_count()

# All the cuSOLVERMp process grids with one slot per device.
PROCESS_GRIDS = [
    (N_DEVICES // c, c) for c in range(1, N_DEVICES + 1) if N_DEVICES % c == 0
]


@requires_gpu
def test_cholesky_distributed_basic():
    """Test cholesky_distributed solver on a small system."""
    # Create a simple positive definite matrix and vector
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(42)
    n = 16 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1  # Make it positive definite
    b = jax.random.normal(key, (n,))

    # Test the solver
    solver = nk.optimizer.solver.cholesky_distributed(local_tile_size=8)
    x, info = solver(A, b)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Solution is not accurate"

    # Check return type
    assert info is None or isinstance(info, dict)


@requires_gpu
def test_cholesky_distributed_vs_cholesky():
    """Test that cholesky_distributed gives same result as standard cholesky."""
    pytest.importorskip("jaxmg")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(123)
    n = 32 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Solve with standard cholesky
    x_standard, _ = nk.optimizer.solver.cholesky(A, b)

    # Solve with distributed cholesky
    x_distributed, _ = nk.optimizer.solver.cholesky_distributed(
        A, b, local_tile_size=16
    )

    # Results should be very similar
    assert jnp.allclose(
        x_standard, x_distributed, rtol=1e-5, atol=1e-5
    ), "Distributed and standard cholesky give different results"


@requires_gpu
def test_cholesky_distributed_with_sharding():
    """Test cholesky_distributed with sharded arrays."""
    pytest.importorskip("jaxmg")

    if N_DEVICES < 2:
        pytest.skip("Need at least 2 devices for sharding test")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(456)
    n = 32 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Shard the arrays as NetKet does for QGT/NTK matrices
    mesh = jax.sharding.get_abstract_mesh()
    A_sharded = jax.device_put(A, jax.sharding.NamedSharding(mesh, P("S", None)))
    b_sharded = jax.device_put(b, jax.sharding.NamedSharding(mesh, P()))

    solver = nk.optimizer.solver.cholesky_distributed(local_tile_size=8)
    x, info = solver(A_sharded, b_sharded)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Sharded solution is not accurate"


@requires_gpu
def test_cholesky_distributed_tiling():
    """Test different tiling sizes."""
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(789)
    n = 64 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Test with different tile sizes
    for tile_size in [None, 16, 32]:
        solver = nk.optimizer.solver.cholesky_distributed(local_tile_size=tile_size)
        x, _ = solver(A, b)

        residual = A @ x - b
        assert (
            jnp.linalg.norm(residual) < 1e-5
        ), f"Solution not accurate with local_tile_size={tile_size}"


@requires_gpu
def test_pinv_smooth_distributed_basic():
    """Test pinv_smooth_distributed solver on a small system."""
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(42)
    n = 16 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1  # Make it positive definite
    b = jax.random.normal(key, (n,))

    # Test the solver
    solver = nk.optimizer.solver.pinv_smooth_distributed(
        local_tile_size=8, rtol=1e-14, rtol_smooth=1e-14
    )
    x, info = solver(A, b)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Solution is not accurate"

    # Check return type
    assert info is None or isinstance(info, dict)


@requires_gpu
def test_pinv_smooth_distributed_vs_pinv_smooth():
    """Test that pinv_smooth_distributed gives same result as standard pinv_smooth."""
    pytest.importorskip("jaxmg")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(123)
    n = 32 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Solve with standard pinv_smooth
    x_standard, _ = nk.optimizer.solver.pinv_smooth(A, b, rtol=1e-12, rtol_smooth=1e-12)

    # Solve with distributed pinv_smooth
    x_distributed, _ = nk.optimizer.solver.pinv_smooth_distributed(
        A, b, local_tile_size=16, rtol=1e-12, rtol_smooth=1e-12
    )

    # Results should be very similar
    assert jnp.allclose(
        x_standard, x_distributed, rtol=1e-5, atol=1e-5
    ), "Distributed and standard pinv_smooth give different results"


@requires_gpu
def test_pinv_smooth_distributed_with_sharding():
    """Test pinv_smooth_distributed with sharded arrays."""
    pytest.importorskip("jaxmg")

    if N_DEVICES < 2:
        pytest.skip("Need at least 2 devices for sharding test")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(456)
    n = 32 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Shard the arrays as NetKet does for QGT/NTK matrices
    mesh = jax.sharding.get_abstract_mesh()
    A_sharded = jax.device_put(A, jax.sharding.NamedSharding(mesh, P("S", None)))
    b_sharded = jax.device_put(b, jax.sharding.NamedSharding(mesh, P()))

    solver = nk.optimizer.solver.pinv_smooth_distributed(
        local_tile_size=8,
        rtol=1e-14,
        rtol_smooth=1e-14,
    )
    x, info = solver(A_sharded, b_sharded)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Sharded solution is not accurate"


@requires_gpu
def test_pinv_smooth_distributed_tiling():
    """Test different tiling sizes for pinv_smooth_distributed."""
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(789)
    n = 64 * N_DEVICES
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Test with different tile sizes
    for tile_size in [None, 16, 32]:
        solver = nk.optimizer.solver.pinv_smooth_distributed(
            local_tile_size=tile_size, rtol=1e-14, rtol_smooth=1e-14
        )
        x, _ = solver(A, b)

        residual = A @ x - b
        assert (
            jnp.linalg.norm(residual) < 1e-5
        ), f"Solution not accurate with local_tile_size={tile_size}"


@requires_gpu
def test_pinv_smooth_distributed_regularization():
    """Test that regularization parameters work correctly."""
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(999)
    n = 32 * N_DEVICES
    # Create a matrix with eigenvalues spanning many orders of magnitude, so
    # that some fall below the cutoff of the more regularized solver.
    Q, _ = jnp.linalg.qr(jax.random.normal(key, (n, n)))
    A = (Q * jnp.logspace(-12, 0, n)) @ Q.T
    b = jax.random.normal(key, (n,))

    # Test with different regularization parameters
    # Higher rtol should give more regularized (smoother) solution
    solver_low = nk.optimizer.solver.pinv_smooth_distributed(
        local_tile_size=16, rtol=1e-16, rtol_smooth=1e-16
    )
    x_low, _ = solver_low(A, b)

    solver_high = nk.optimizer.solver.pinv_smooth_distributed(
        local_tile_size=16, rtol=1e-6, rtol_smooth=1e-6
    )
    x_high, _ = solver_high(A, b)

    # Solutions should differ due to different regularization
    # (but both should still be valid solutions, just with different conditioning)
    assert not jnp.allclose(
        x_low, x_high, rtol=1e-3
    ), "Different regularization should give different solutions"


### Tests of the interface with jaxmg, which do not require jaxmg nor a GPU.
#
# jaxmg's cuSOLVERMp kernels only run on GPUs, with one process per GPU, so the
# tests above cannot run in CI. The tests below install a fake `jaxmg` module
# that solves the system with plain jax, and check that NetKet hands it inputs
# satisfying cuSOLVERMp's requirements.


def _fake_place(x, sharding):
    if all(t == AxisType.Explicit for t in sharding.mesh.axis_types):
        return jax.reshard(x, sharding)
    return jax.lax.with_sharding_constraint(x, sharding)


def _fake_infer_layout(a):
    """
    Infer the mesh and the matrix sharding of `a` like jaxmg >= 1.4: off its
    sharding, or inside of jit off its type, falling back to the context mesh,
    and to sharding the rows (and columns) over the axes of the mesh.
    """
    sharding = getattr(a, "sharding", None)
    if not isinstance(sharding, NamedSharding):
        sharding = getattr(jax.typeof(a), "sharding", None)
    if not isinstance(sharding, NamedSharding) or sharding.mesh.empty:
        sharding = None
    mesh = jax.sharding.get_abstract_mesh() if sharding is None else sharding.mesh
    if mesh.empty:
        raise ValueError("jaxmg could not find a mesh for A.")
    if sharding is not None and any(axis is not None for axis in sharding.spec):
        specs = sharding.spec
    else:
        specs = P(*mesh.axis_names)
    return mesh, P(*specs, *(None,) * (2 - len(specs)))


_FAKE_STATUS_SIZE = 40


def _fake_status(status_code):
    """Per-process status vectors, concatenated, failing on the last process."""
    status = np.zeros((N_DEVICES, _FAKE_STATUS_SIZE), np.int32)
    status[:, 1:] = 7  # other diagnostic fields are not all zero
    status[-1, 0] = status_code
    return jnp.asarray(status.reshape(-1))


def _install_fake_jaxmg(monkeypatch, version="1.4.1", status_code=0):
    """
    Install a fake `jaxmg` recording its calls and solving with plain jax.

    Like jaxmg >= 1.4, the fake infers the mesh and the sharding of `A` by
    itself, enters that mesh, places `A` in that sharding, and maps it over the
    mesh with :func:`jax.shard_map`, which checks that NetKet laid `A` out on
    that mesh consistently with the context mesh. Its high-level entry points donate their inputs, like
    jaxmg's, while the `_shardmap_ctx` ones do not. With a nonzero
    `status_code`, the fake reports a failure and, like jaxmg >= 1.4.1, fills
    its outputs with NaNs.
    """
    calls = []

    def _layout(a, T_A):
        mesh, specs = _fake_infer_layout(a)
        calls.append({"n": a.shape[0], "T_A": T_A, "mesh": mesh, "specs": specs})
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            a = _fake_place(a, NamedSharding(mesh, specs))
            a = jax.shard_map(lambda x: x, mesh=mesh, in_specs=specs, out_specs=specs)(
                a
            )
            return mesh, specs, _fake_place(a, NamedSharding(mesh, P()))

    def potrs_shardmap_ctx(a, b, T_A, mesh=None, matrix_specs=None, **kwargs):
        assert mesh is None and matrix_specs is None
        if a.dtype != b.dtype:
            raise TypeError("potrs requires A and b to have the same dtype.")
        mesh, _, a = _layout(a, T_A)
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            b = _fake_place(b, NamedSharding(mesh, P()))
            x = jnp.linalg.solve(a, b)
            if status_code != 0:
                x = jnp.full_like(x, jnp.nan)
        # (a_work, x, status), like jaxmg's non-donating entry point
        return a, x, _fake_status(status_code)

    def syevd_shardmap_ctx(a, T_A, mesh=None, matrix_specs=None, **kwargs):
        assert mesh is None and matrix_specs is None
        mesh, specs, a = _layout(a, T_A)
        with jax.sharding.use_abstract_mesh(mesh.abstract_mesh):
            w, v = jnp.linalg.eigh(a)
            # Like jaxmg, give the eigenvectors back sharded as the matrix.
            v = _fake_place(v, NamedSharding(mesh, specs))
            if status_code != 0:
                w, v = jnp.full_like(w, jnp.nan), jnp.full_like(v, jnp.nan)
        return a, w, v, _fake_status(status_code)

    @partial(jax.jit, static_argnums=2, donate_argnums=(0, 1))
    def potrs(a, b, T_A, **kwargs):
        return jnp.linalg.solve(a, b)

    @partial(jax.jit, static_argnums=1, donate_argnums=0)
    def syevd(a, T_A, **kwargs):
        return jnp.linalg.eigh(a)

    module = ModuleType("jaxmg")
    module.__version__ = version
    module.potrs = potrs
    module.syevd = syevd
    module.potrs_shardmap_ctx = potrs_shardmap_ctx
    module.syevd_shardmap_ctx = syevd_shardmap_ctx
    monkeypatch.setitem(sys.modules, "jaxmg", module)
    return calls


def _check_cusolvermp_contract(call, expected_grid):
    """Check a recorded call against cuSOLVERMp's layout requirements."""
    mesh, specs, n, T_A = call["mesh"], call["specs"], call["n"], call["T_A"]

    # One slot of the process grid per device.
    assert mesh.size == N_DEVICES

    # Each matrix dimension is either mapped onto one axis of the mesh, or not
    # distributed (a degenerate grid), and the grid has the expected shape.
    assert isinstance(specs, P) and len(specs) == 2
    assert any(axis is not None for axis in specs)
    assert all(axis is None or axis in mesh.axis_names for axis in specs)
    grid = tuple(1 if axis is None else mesh.shape[axis] for axis in specs)
    assert grid == expected_grid
    assert mesh.size == grid[0] * grid[1]

    process_rows, process_cols = expected_grid
    # The matrix must be evenly block-distributed over the grid...
    assert n % process_rows == 0
    assert n % process_cols == 0
    # ... and every process row/column must own at least one tile.
    assert T_A > 0
    assert -(-n // T_A) >= max(process_rows, process_cols)
    # The default tile size avoids padding the local shards.
    assert (n // process_rows) % T_A == 0
    assert (n // process_cols) % T_A == 0


@pytest.fixture(params=[AxisType.Auto, AxisType.Explicit], ids=["auto", "explicit"])
def context_mesh(request):
    """Run the test with a NetKet-like 'S' mesh of the given axis type."""
    mesh = Mesh(np.asarray(jax.devices()), ("S",), axis_types=(request.param,))
    with jax.set_mesh(mesh):
        yield mesh


def _spd_problem(n, context_mesh, A_specs=P("S", None)):
    key = jax.random.PRNGKey(42)
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T / n + jnp.eye(n)
    b = jax.random.normal(key, (n,))
    A_sharded = jax.device_put(A, NamedSharding(context_mesh, A_specs))
    b_sharded = jax.device_put(b, NamedSharding(context_mesh, P()))
    return A, b, A_sharded, b_sharded


@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize("process_grid", [None, *PROCESS_GRIDS])
@pytest.mark.parametrize(
    "A_specs",
    [P("S", None), P(None, "S"), P()],
    ids=["A_rows", "A_cols", "A_replicated"],
)
@pytest.mark.parametrize(
    "solver, reference",
    [
        pytest.param(
            nk.optimizer.solver.cholesky_distributed,
            nk.optimizer.solver.cholesky,
            id="cholesky",
        ),
        pytest.param(
            nk.optimizer.solver.pinv_smooth_distributed,
            nk.optimizer.solver.pinv_smooth,
            id="pinv_smooth",
        ),
    ],
)
def test_distributed_solvers_interface(
    monkeypatch, context_mesh, solver, reference, A_specs, process_grid, jit
):
    calls = _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    A, b, A_sharded, b_sharded = _spd_problem(n, context_mesh, A_specs)

    def solve(A, b):
        return solver(A, b, process_grid=process_grid)

    if jit:
        solve = jax.jit(solve)
    x, info = solve(A_sharded, b_sharded)

    assert info is None
    # The solution is on the context mesh, where NetKet can use it, and with
    # Explicit mesh axes it is replicated like `b`.
    assert x.sharding.mesh.axis_names == context_mesh.axis_names
    if context_mesh.axis_types[0] == AxisType.Explicit:
        assert x.sharding.is_fully_replicated
    np.testing.assert_allclose(x, reference(A, b)[0], rtol=1e-5, atol=1e-8)

    assert len(calls) == 1
    if process_grid is not None:
        expected_grid = process_grid
    elif A_specs == P(None, "S") and (
        context_mesh.axis_types[0] == AxisType.Explicit or not jit
    ):
        # By default, jaxmg follows the sharding of `A`, where it is known.
        expected_grid = (1, N_DEVICES)
    else:
        expected_grid = (N_DEVICES, 1)
    _check_cusolvermp_contract(calls[0], expected_grid)
    if process_grid is None:
        # The default grid lives on the context mesh, so `A` is not moved.
        assert calls[0]["mesh"].axis_names == context_mesh.axis_names


@pytest.mark.parametrize(
    "n_local, max_tile_size, expected",
    [
        (2048, 4096, 2048),  # the whole local shard
        (12500, 4096, 3125),  # the largest divisor below the cap
        (12500, 512, 500),
        # prime: padding beats tiny tiles, but is kept to a minimum
        (4099, 512, 456),
        (4099, 4096, 2050),
        (40, 512, 40),  # small matrices use a single tile
    ],
)
def test_default_tile_size(n_local, max_tile_size, expected):
    n = n_local * N_DEVICES
    assert default_tile_size(n, (N_DEVICES, 1), max_tile_size=max_tile_size) == expected


def test_cholesky_distributed_mixed_dtypes(monkeypatch, context_mesh):
    """jaxmg requires A and b to have the same dtype, unlike jnp.linalg.solve."""
    _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    A, b, A_sharded, b_sharded = _spd_problem(n, context_mesh)
    b_sharded = b_sharded.astype(jnp.float32)

    x, _ = nk.optimizer.solver.cholesky_distributed(A_sharded, b_sharded)
    assert x.dtype == A.dtype
    np.testing.assert_allclose(x, nk.optimizer.solver.cholesky(A, b)[0], rtol=1e-5)


@pytest.mark.skipif(N_DEVICES == 1, reason="requires more than one device")
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
def test_distributed_solvers_reordered_mesh(monkeypatch, solver, jit):
    """
    jaxmg resolves the devices of the process grid at run time, so on the
    default grid the devices of the mesh need not be in `jax.devices()` order.
    The dedicated mesh of other grids uses that order, which jax requires the
    context mesh to share.
    """
    _install_fake_jaxmg(monkeypatch)

    mesh = Mesh(np.asarray(jax.devices())[::-1], ("S",))
    n = 16 * N_DEVICES
    with jax.set_mesh(mesh):
        A, b, A_sharded, b_sharded = _spd_problem(n, mesh)

        def solve(A, b, process_grid=None):
            fn = partial(solver, process_grid=process_grid)
            return (jax.jit(fn) if jit else fn)(A, b)

        x, _ = solve(A_sharded, b_sharded)
        assert x.sharding.mesh == mesh
        np.testing.assert_allclose(x, np.linalg.solve(A, b), rtol=1e-5, atol=1e-8)

        with pytest.raises(ValueError, match="incompatible devices"):
            solve(A_sharded, b_sharded, process_grid=PROCESS_GRIDS[-1])


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
def test_invalid_process_grid(monkeypatch, solver):
    _install_fake_jaxmg(monkeypatch)

    A = jnp.eye(16)
    b = jnp.ones((16,))

    with pytest.raises(ValueError, match="one slot per device"):
        solver(A, b, process_grid=(N_DEVICES + 1, 2))

    with pytest.raises(ValueError, match="must be a pair"):
        solver(A, b, process_grid=(N_DEVICES,))


@pytest.mark.skipif(N_DEVICES == 1, reason="requires more than one device")
@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
def test_matrix_size_not_divisible_by_grid(monkeypatch, solver):
    """cuSOLVERMp needs every process to own the same number of rows/columns."""
    _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES + 1
    A = jnp.eye(n)
    b = jnp.ones((n,))

    with pytest.raises(ValueError, match="divisible by both grid dimensions"):
        solver(A, b)


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
@pytest.mark.parametrize("version", ["0.0.9", "1.0.0", "1.1.0", "1.3.0", "1.4.0"])
def test_unsupported_jaxmg_version(monkeypatch, solver, version):
    """jaxmg 0.0.x wrapped cuSOLVERMg through a different, unsupported API, and
    jaxmg < 1.4 cannot infer the mesh and the sharding of `A` inside of jit, and
    jaxmg < 1.4.1 can return finite but wrong output when the solver fails."""
    _install_fake_jaxmg(monkeypatch, version=version)

    A = jnp.eye(16)
    b = jnp.ones((16,))

    with pytest.raises(ImportError, match=rf"jaxmg.*{re.escape(version)}"):
        solver(A, b)


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
def test_distributed_solvers_do_not_consume_the_problem(monkeypatch, solver):
    """`nan_fallback` reuses A and b, so the solvers must not donate them."""
    _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    A = jnp.eye(n) * 2.0
    b = jnp.ones((n,))

    combined = nk.optimizer.solver.nan_fallback(solver, nk.optimizer.solver.cholesky)
    x, _ = combined(A, b)

    np.testing.assert_allclose(x, jnp.full((n,), 0.5), rtol=1e-5)
    assert not A.is_deleted()
    assert not b.is_deleted()


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
@pytest.mark.parametrize("jit", [False, True], ids=["eager", "jit"])
def test_distributed_solvers_report_backend_failures(
    monkeypatch, context_mesh, solver, jit
):
    """
    jaxmg returns NaNs when the native solver fails, e.g. for a singular matrix.
    The solvers must pass them on, so that `nan_fallback` falls back.
    """
    _install_fake_jaxmg(monkeypatch, status_code=26)

    n = 16 * N_DEVICES
    # Replicated, as the dense fallback does not accept a sharded `A` with
    # Explicit mesh axes.
    A = jax.device_put(
        jnp.diag(jnp.full(n, 2.0).at[-1].set(0.0)), NamedSharding(context_mesh, P())
    )
    b = jax.device_put(jnp.ones(n).at[-1].set(0.0), NamedSharding(context_mesh, P()))

    x, _ = (jax.jit(solver) if jit else solver)(A, b)
    assert np.all(np.isnan(x))

    combined = nk.optimizer.solver.nan_fallback(solver, nk.optimizer.solver.pinv_smooth)
    x, info = (jax.jit(combined) if jit else combined)(A, b)
    assert info["solver_fallback"]
    np.testing.assert_allclose(x, np.where(np.arange(n) < n - 1, 0.5, 0.0), atol=1e-8)
