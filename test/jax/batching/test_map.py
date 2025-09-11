import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

from netket.jax.lax import map as nkmap, apply as nkapply

import pytest

from test.common_mesh import with_explicit_meshes


def f1(W, x, check_ndim=None):
    if check_ndim is not None:
        assert (
            x.ndim == check_ndim
        ), f"Expected x to have {check_ndim} dimensions, got {x.ndim}"
    return jnp.sum(x @ W, axis=-1)


def f2(W, x, check_ndim=None):
    if check_ndim is not None:
        assert (
            x.ndim == check_ndim
        ), f"Expected x to have {check_ndim} dimensions, got {x.ndim}"
    return x @ W


@pytest.mark.parametrize("fun", [f1, f2])
@pytest.mark.parametrize(
    "jit", [pytest.param(True, id="jit"), pytest.param(False, id="no_jit")]
)
@with_explicit_meshes(
    [
        None,
        ((2,), ("S",)),
        # ((1,2), ("S","P")), # broken because of jax-ml/jax#29635
        ((2, 2), ("S", "P")),
    ]
)
def test_map(fun, mesh, jit):
    Ns = 32
    N = 4
    xs = jnp.zeros((Ns, N))
    W = jnp.ones((N, N))

    def fun_map(W, x, batch_size=None, check_ndim=1):
        return nkmap(lambda x: fun(W, x, check_ndim=1), xs, batch_size=batch_size)

    if jit:
        fun_map = jax.jit(fun_map, static_argnames=("batch_size",))

    result = jax.vmap(fun, in_axes=(None, 0), out_axes=0)(W, xs)
    scan_result = fun_map(W, xs)
    scan_batch4_result = fun_map(W, xs, batch_size=4)
    scan_batch5_result = fun_map(W, xs, batch_size=5)

    assert jnp.allclose(result, scan_result)
    assert result.sharding == scan_result.sharding
    assert jnp.allclose(result, scan_batch4_result)
    assert result.sharding == scan_batch4_result.sharding
    assert jnp.allclose(result, scan_batch5_result)
    assert result.sharding == scan_batch5_result.sharding

    if mesh.empty:
        return
    if len(mesh.axis_names) >= 1:
        xs = reshard(xs, P(mesh.axis_names[0]))
    if len(mesh.axis_names) >= 2:
        W = reshard(W, P(None, mesh.axis_names[1]))
    # sharding

    sharded_result = jax.vmap(fun, in_axes=(None, 0), out_axes=0)(W, xs)
    scan_result = fun_map(W, xs)
    scan_batch4_result = fun_map(W, xs, batch_size=4)
    scan_batch5_result = fun_map(W, xs, batch_size=5)

    assert jnp.allclose(sharded_result, scan_result)
    assert sharded_result.sharding == scan_result.sharding
    assert jnp.allclose(sharded_result, scan_batch4_result)
    assert sharded_result.sharding == scan_batch4_result.sharding
    assert jnp.allclose(sharded_result, scan_batch5_result)
    assert sharded_result.sharding == scan_batch5_result.sharding

    # Checlk that different lengths raise an error
    with pytest.raises(ValueError, match=r".*different leading axis sizes:.*"):
        scan_result = nkmap(lambda args: fun(*args), (W, xs))


@pytest.mark.parametrize("fun", [f1, f2])
@pytest.mark.parametrize(
    "jit", [pytest.param(True, id="jit"), pytest.param(False, id="no_jit")]
)
@with_explicit_meshes(
    [
        None,
        ((2,), ("S",)),
        # ((1,2), ("S","P")), # broken because of jax-ml/jax#29635
        ((2, 2), ("S", "P")),
    ]
)
def test_apply(fun, mesh, jit):
    Ns = 32
    N = 4
    xs = jnp.zeros((Ns, N))
    W = jnp.ones((N, N))

    def fun_apply(W, x, batch_size=None, check_ndim=2):
        return nkapply(
            lambda x: fun(W, x, check_ndim=check_ndim), xs, batch_size=batch_size
        )

    if jit:
        fun_apply = jax.jit(fun_apply, static_argnames=("batch_size", "check_ndim"))

    result = jax.vmap(fun, in_axes=(None, 0), out_axes=0)(W, xs)
    scan_result = fun_apply(W, xs)
    scan_batch4_result = fun_apply(W, xs, batch_size=4)
    scan_batch5_result = fun_apply(W, xs, batch_size=5)

    assert jnp.allclose(result, scan_result)
    assert result.sharding == scan_result.sharding
    assert jnp.allclose(result, scan_batch4_result)
    assert result.sharding == scan_batch4_result.sharding
    assert jnp.allclose(result, scan_batch5_result)
    assert result.sharding == scan_batch5_result.sharding

    if mesh.empty:
        return
    if len(mesh.axis_names) >= 1:
        xs = reshard(xs, P(mesh.axis_names[0]))
    if len(mesh.axis_names) >= 2:
        W = reshard(W, P(None, mesh.axis_names[1]))
    # sharding

    sharded_result = jax.vmap(fun, in_axes=(None, 0), out_axes=0)(W, xs)
    scan_result = fun_apply(W, xs)
    scan_batch4_result = fun_apply(W, xs, batch_size=4)
    scan_batch5_result = fun_apply(W, xs, batch_size=5)

    assert jnp.allclose(sharded_result, scan_result)
    assert sharded_result.sharding == scan_result.sharding
    assert jnp.allclose(sharded_result, scan_batch4_result)
    assert sharded_result.sharding == scan_batch4_result.sharding
    assert jnp.allclose(sharded_result, scan_batch5_result)
    assert sharded_result.sharding == scan_batch5_result.sharding

    # Checlk that different lengths raise an error
    with pytest.raises(ValueError, match=r".*different leading axis sizes:.*"):
        scan_result = nkmap(lambda args: fun(*args), (W, xs))
