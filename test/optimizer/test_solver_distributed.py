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

import sys
from types import ModuleType

import numpy as np
import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import netket as nk

from test import common  # noqa: F401

# jaxmg's FFI handlers (potrs_mg, syevd_mg) are only registered for the CUDA
# platform. Even when jaxmg is installed (e.g. on Linux CI), the kernels cannot
# run on a CPU/Host backend, so these tests must be skipped unless a GPU is
# available.
requires_gpu = pytest.mark.skipif(
    jax.default_backend() != "gpu",
    reason="jaxmg distributed solvers require a GPU backend",
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
    # Create a matrix with some small eigenvalues
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 1e-8  # Small regularization
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


def _install_fake_jaxmg(monkeypatch, version="1.0.0"):
    """Install a fake `jaxmg` recording its calls and solving with plain jax."""
    calls = []

    def potrs(a, b, T_A, mesh=None, matrix_specs=None, **kwargs):
        calls.append({"n": a.shape[0], "T_A": T_A, "mesh": mesh, "specs": matrix_specs})
        return jnp.linalg.solve(a, b)

    def syevd(a, T_A, mesh=None, matrix_specs=None, **kwargs):
        calls.append({"n": a.shape[0], "T_A": T_A, "mesh": mesh, "specs": matrix_specs})
        return jnp.linalg.eigh(a)

    module = ModuleType("jaxmg")
    module.__version__ = version
    module.potrs = potrs
    module.syevd = syevd
    monkeypatch.setitem(sys.modules, "jaxmg", module)
    return calls


def _check_cusolvermp_contract(call, expected_grid):
    """Check a recorded call against cuSOLVERMp's layout requirements."""
    mesh, specs, n, T_A = call["mesh"], call["specs"], call["n"], call["T_A"]

    # The process grid must be 2D, with one slot per device.
    assert isinstance(mesh, Mesh)
    assert mesh.devices.shape == expected_grid
    assert sorted(d.id for d in mesh.devices.flat) == sorted(
        d.id for d in jax.devices()
    )

    # Both matrix dimensions must be mapped to a named mesh axis.
    assert isinstance(specs, P)
    assert tuple(specs) == tuple(mesh.axis_names)

    process_rows, process_cols = expected_grid
    # The matrix must be evenly block-distributed over the grid...
    assert n % process_rows == 0
    assert n % process_cols == 0
    # ... and every process row/column must own at least one tile.
    assert T_A > 0
    assert -(-n // T_A) >= max(process_rows, process_cols)


@pytest.mark.parametrize("process_grid", [None, *PROCESS_GRIDS])
def test_cholesky_distributed_interface(monkeypatch, process_grid):
    calls = _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    key = jax.random.PRNGKey(42)
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    x, info = nk.optimizer.solver.cholesky_distributed(A, b, process_grid=process_grid)

    assert info is None
    np.testing.assert_allclose(x, nk.optimizer.solver.cholesky(A, b)[0], rtol=1e-5)

    assert len(calls) == 1
    expected_grid = (N_DEVICES, 1) if process_grid is None else process_grid
    _check_cusolvermp_contract(calls[0], expected_grid)


@pytest.mark.parametrize("process_grid", [None, *PROCESS_GRIDS])
def test_pinv_smooth_distributed_interface(monkeypatch, process_grid):
    calls = _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    key = jax.random.PRNGKey(42)
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    x, info = nk.optimizer.solver.pinv_smooth_distributed(
        A, b, process_grid=process_grid
    )

    assert info is None
    np.testing.assert_allclose(x, nk.optimizer.solver.pinv_smooth(A, b)[0], rtol=1e-5)

    assert len(calls) == 1
    expected_grid = (N_DEVICES, 1) if process_grid is None else process_grid
    _check_cusolvermp_contract(calls[0], expected_grid)


@pytest.mark.parametrize(
    "solver",
    [
        pytest.param(nk.optimizer.solver.cholesky_distributed, id="cholesky"),
        pytest.param(nk.optimizer.solver.pinv_smooth_distributed, id="pinv_smooth"),
    ],
)
def test_distributed_solvers_inside_jit(monkeypatch, solver):
    """The solvers are usually called from inside of `jax.jit`."""
    _install_fake_jaxmg(monkeypatch)

    n = 16 * N_DEVICES
    A = jnp.eye(n) * 2.0
    b = jnp.ones((n,))

    x = jax.jit(lambda A, b: solver(A, b)[0])(A, b)
    np.testing.assert_allclose(x, jnp.full((n,), 0.5), rtol=1e-5)


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
def test_unsupported_jaxmg_version(monkeypatch, solver):
    """jaxmg 0.0.x wrapped cuSOLVERMg through a different, unsupported API."""
    _install_fake_jaxmg(monkeypatch, version="0.0.9")

    A = jnp.eye(16)
    b = jnp.ones((16,))

    with pytest.raises(ImportError, match=r"jaxmg.*0\.0\.9"):
        solver(A, b)
