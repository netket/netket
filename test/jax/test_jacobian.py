import pytest

import jax
import jax.numpy as jnp
import numpy as np

from flax import core as fcore

import netket as nk

from .. import common


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


@common.onlyif_sharding
@pytest.mark.parametrize("sharded", [False, True])
def test_real_function_sharding(sharded):
    k = jax.random.key(1)

    ma = nk.models.RBM(param_dtype=jnp.float32)

    if sharded:
        n_samples = 2 * jax.device_count()
    else:
        n_samples = 3

    xs = jax.random.normal(k, (n_samples, 4))
    if sharded:
        xs = jax.lax.with_sharding_constraint(
            xs, jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), jax.P("S"))
        )

    model_state, parameters = fcore.pop(ma.init(k, xs), "params")

    def afun(vars, xs):
        return ma.apply(vars, xs) + 0.3j * ma.apply(vars, xs)

    jac_re = nk.jax.jacobian(
        afun,
        parameters,
        xs,
        model_state,
        mode="real",
        dense=True,
        _axis_0_is_sharded=sharded,
    )

    def afun(vars, xs):
        return ma.apply(vars, xs)

    jac_2 = nk.jax.jacobian(
        afun,
        parameters,
        xs,
        model_state,
        mode="real",
        dense=True,
        _axis_0_is_sharded=sharded,
    )

    if sharded:
        assert jac_re.sharding.spec == jax.P("S")
        assert jac_2.sharding.spec == jax.P("S")
        jac_re = jax.lax.with_sharding_constraint(
            jac_re,
            jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), jax.P()),
        )
        jac_2 = jax.lax.with_sharding_constraint(
            jac_2, jax.sharding.NamedSharding(jax.sharding.get_abstract_mesh(), jax.P())
        )

    np.testing.assert_allclose(jac_re, jac_2)
