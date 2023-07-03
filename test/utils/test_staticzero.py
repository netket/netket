from netket.utils.numbers import StaticZero
import numpy as np

import jax
import jax.numpy as jnp


def test_falseness():
    a = StaticZero()
    b = StaticZero()

    assert a == b
    assert hash(a) == hash(b)
    assert not a


def test_jax_pytree():
    a = StaticZero()

    data, struct = jax.tree_util.tree_flatten(a)
    b = jax.tree_util.tree_unflatten(struct, data)

    assert a == b
    assert len(data) == 0


def test_array_interface():
    a = StaticZero()

    assert a.shape == ()
    assert a.ndim == 0
    assert a.weak_dtype
    assert isinstance(a.astype(float), StaticZero)
    assert a.astype(float).dtype == float

    np.testing.assert_allclose(np.asarray(a), np.array(False))
    np.testing.assert_allclose(jnp.asarray(a), np.array(False))


def test_binary_ops():
    a = StaticZero()
    a2 = StaticZero()

    assert -a == a
    assert a + a2 == a
    assert a - a2 == a
    assert a * a2 == a

    assert a + 2.0 == 2.0
    assert 2.0 + a == 2.0
    assert a - 2.0 == -2.0
    assert 2.0 - a == 2.0
    assert a * 2.0 == a
    assert 2.0 * a == a
