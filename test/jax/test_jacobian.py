import pytest

import jax
import jax.numpy as jnp
from jax.sharding import reshard, PartitionSpec as P

import numpy as np

from flax import core as fcore

import netket as nk

from test import common, common_mesh


@common.skipif_distributed
def test_real_function_error():
    k = jax.random.key(1)

    ma = nk.models.RBM(param_dtype=jnp.float32)
    xs = jax.random.normal(k, (3, 4))
    model_state, parameters = fcore.pop(ma.init(k, xs), "params")

    def afun(vars, xs):
        return ma.apply(vars, xs)

    with pytest.raises(TypeError, match="Cannot build the complex"):
        nk.jax.jacobian(afun, parameters, xs, model_state, mode="complex")


@common.skipif_distributed
def test_real_function_output():
    k = jax.random.key(1)

    ma = nk.models.RBM(param_dtype=jnp.float32)
    xs = jax.random.normal(k, (3, 4))
    model_state, parameters = fcore.pop(ma.init(k, xs), "params")

    def afun(vars, xs):
        return ma.apply(vars, xs) + 0.3j * ma.apply(vars, xs)

    jac_re = nk.jax.jacobian(afun, parameters, xs, model_state, mode="real", dense=True)

    def afun(vars, xs):
        return ma.apply(vars, xs)

    jac_2 = nk.jax.jacobian(afun, parameters, xs, model_state, mode="real", dense=True)

    np.testing.assert_allclose(jac_re, jac_2)


@common_mesh.with_explicit_meshes([None, ((2,), ("S"))])
@pytest.mark.parametrize("sharded", [False, True])
def test_real_function_sharding(mesh, sharded):
    k = jax.random.key(1)

    ma = nk.models.RBM(param_dtype=jnp.float32)

    if mesh.empty and sharded:
        # thist est should not exist
        return
    elif not mesh.empty and sharded:
        n_samples = 2 * jax.device_count()
    else:
        n_samples = 3

    xs = jax.random.normal(k, (n_samples, 4))
    if not mesh.empty and sharded:
        xs = reshard(xs, P("S"))

    model_state, parameters = fcore.pop(ma.init(k, xs), "params")

    jac_re = nk.jax.jacobian(
        lambda vars, xs: ma.apply(vars, xs) + 0.3j * ma.apply(vars, xs),
        parameters,
        xs,
        model_state,
        mode="real",
        dense=True,
    )

    jac_2 = nk.jax.jacobian(
        ma.apply,
        parameters,
        xs,
        model_state,
        mode="real",
        dense=True,
    )

    if not mesh.empty:
        spec = "S" if sharded else None
        assert jac_re.sharding.spec == P(spec, None)
        assert jac_2.sharding.spec == P(spec, None)

    np.testing.assert_allclose(jac_re, jac_2)
