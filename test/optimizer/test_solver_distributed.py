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

"""Tests for distributed solvers (cholesky_distributed)."""

import pytest
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P

import netket as nk

from test import common  # noqa: F401


def test_cholesky_distributed_basic():
    """Test cholesky_distributed solver on a small system."""
    # Create a simple positive definite matrix and vector
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(42)
    n = 16
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1  # Make it positive definite
    b = jax.random.normal(key, (n,))

    # Test the solver
    solver = nk.optimizer.solver.cholesky_distributed(T_A=8)
    x, info = solver(A, b)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Solution is not accurate"

    # Check return type
    assert info is None or isinstance(info, dict)


def test_cholesky_distributed_vs_cholesky():
    """Test that cholesky_distributed gives same result as standard cholesky."""
    pytest.importorskip("jaxmg")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(123)
    n = 32
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Solve with standard cholesky
    x_standard, _ = nk.optimizer.solver.cholesky(A, b)

    # Solve with distributed cholesky
    x_distributed, _ = nk.optimizer.solver.cholesky_distributed(A, b, T_A=16)

    # Results should be very similar
    assert jnp.allclose(
        x_standard, x_distributed, rtol=1e-5, atol=1e-5
    ), "Distributed and standard cholesky give different results"


def test_cholesky_distributed_with_sharding():
    """Test cholesky_distributed with sharded arrays."""
    pytest.importorskip("jaxmg")

    ndevices = len(jax.devices())
    if ndevices < 2:
        pytest.skip("Need at least 2 devices for sharding test")

    # Create a simple positive definite matrix and vector
    key = jax.random.PRNGKey(456)
    n = 32
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Create mesh
    devices = jax.devices()[: min(ndevices, 4)]
    mesh = Mesh(devices, axis_names=("S",))

    # Shard the arrays
    sharding_A = jax.sharding.NamedSharding(mesh, P("S", None))
    sharding_b = jax.sharding.NamedSharding(mesh, P())

    A_sharded = jax.device_put(A, sharding_A)
    b_sharded = jax.device_put(b, sharding_b)

    # Test the solver with explicit mesh
    solver = nk.optimizer.solver.cholesky_distributed(
        T_A=8, mesh=mesh, in_specs=(P("S", None), P(None, None))
    )

    with mesh:
        x, info = solver(A_sharded, b_sharded)

    # Verify the solution
    residual = A @ x - b
    assert jnp.linalg.norm(residual) < 1e-5, "Sharded solution is not accurate"


def test_cholesky_distributed_tiling():
    """Test different tiling sizes."""
    pytest.importorskip("jaxmg")

    key = jax.random.PRNGKey(789)
    n = 64
    A_base = jax.random.normal(key, (n, n))
    A = A_base @ A_base.T + jnp.eye(n) * 0.1
    b = jax.random.normal(key, (n,))

    # Test with different tile sizes
    for T_A in [None, 16, 32]:
        solver = nk.optimizer.solver.cholesky_distributed(T_A=T_A)
        x, _ = solver(A, b)

        residual = A @ x - b
        assert jnp.linalg.norm(residual) < 1e-5, f"Solution not accurate with T_A={T_A}"
