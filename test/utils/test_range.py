import pytest

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils import StaticRange

from .. import common

pytestmark = common.skipif_mpi


def test_staticrange_eq():
    ran = StaticRange(0, 10, 100)

    assert hash(StaticRange(0, 10, 100)) == hash(StaticRange(0, 10, 100))
    assert StaticRange(0, 10, 100) == StaticRange(0, 10, 100)

    assert hash(StaticRange(0, 10, 100)) != hash(StaticRange(0, 11, 10))
    assert StaticRange(0, 10, 100) != StaticRange(0, 11, 100)

    assert hash(StaticRange(0, 10, 100, dtype=int)) != hash(
        StaticRange(0, 11, 10, dtype=float)
    )
    assert StaticRange(0, 10, 100) != StaticRange(0, 11, 100)

    leaves, treedef1 = jax.tree_util.tree_flatten(ran)
    _, treedef2 = jax.tree_util.tree_flatten(ran)
    assert len(leaves) == 0
    assert hash(treedef1) == hash(treedef2)

    assert ran == jax.tree_util.tree_unflatten(treedef1, leaves)

    isinstance(repr(ran), str)


def test_staticrange_array_interface():
    ran = StaticRange(0, 10, 100)

    assert ran.dtype == int
    assert ran.ndim == 1
    assert ran.shape == (100,)
    assert len(ran) == 100
    np.testing.assert_allclose(np.array(ran), np.arange(0, 1000, 10))

    assert StaticRange(0, 10, 100, dtype=float).astype(int) == ran

    assert ran[10] == 10 * 10
    with pytest.raises(IndexError):
        ran[101]

    np.testing.assert_allclose(
        ran.states_to_numbers(np.array([0, 10, 100])), np.array([0, 1, 10])
    )
    assert ran.states_to_numbers(10).dtype == int
    assert ran.states_to_numbers(10, dtype=float).dtype == float
    assert isinstance(ran.states_to_numbers(10), np.ndarray)
    assert isinstance(ran.states_to_numbers(np.array([10, 20])), np.ndarray)
    assert isinstance(ran.states_to_numbers(jnp.array(10)), jax.Array)

    np.testing.assert_allclose(
        ran.numbers_to_states(np.array([0, 1, 10])), np.array([0, 10, 100])
    )
    assert ran.numbers_to_states(1).dtype == int
    assert isinstance(ran.numbers_to_states(1), np.int_)
    assert isinstance(ran.numbers_to_states(np.array([1, 2])), np.ndarray)
    assert isinstance(ran.numbers_to_states(jnp.array(1)), jax.Array)

    ran = StaticRange(0, 10, 100, dtype=float)
    assert ran.dtype == float
    assert ran.states_to_numbers(1).dtype == int
    assert ran.numbers_to_states(1).dtype == float
    assert isinstance(ran.numbers_to_states(1), float)
    assert np.array(ran).dtype == float
    np.testing.assert_allclose(
        np.array(StaticRange(0, 10, 100)),
        np.array(StaticRange(0, 10, 100, dtype=float).astype(int)),
    )


def test_staticrange_flip():
    ran = StaticRange(0, 10, 100, dtype=float)

    with pytest.raises(ValueError):
        ran.flip_state(10)

    ran = StaticRange(0, 1, 2)
    ran.flip_state(0) == 1
