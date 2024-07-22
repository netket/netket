import pytest

import jax
import jax.numpy as jnp
import numpy as np

from flax import core as fcore

import netket as nk

from .. import common


@common.skipif_mpi
def test_real_function_error():
    k = jax.random.key(1)

    ma = nk.models.RBM(param_dtype=jnp.float32)
    xs = jax.random.normal(k, (3, 4))
    model_state, parameters = fcore.pop(ma.init(k, xs), "params")

    def afun(vars, xs):
        return ma.apply(vars, xs)

    with pytest.raises(TypeError, match="Cannot build the complex"):
        nk.jax.jacobian(afun, parameters, xs, model_state, mode="complex")


@common.skipif_mpi
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
