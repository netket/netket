import netket as nk

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass


@dataclass
class MockCompoundType:
    field1: int
    field2: float

    def to_compound(self):
        return "field2", {"field1": self.field1, "field2": self.field2}


def create_mock_data_iter(iter):
    return {
        "int": iter,
        "complex": iter + 1j * iter,
        "npint": np.array(iter),
        "jaxcomplex": jnp.array(iter + 1j * iter),
        "dict": {"int": iter},
        "compound": MockCompoundType(iter, iter * 10),
    }


def test_accum_mvhistory():
    L = 10

    tree = None
    for i in range(L):
        tree = nk.utils.accum_histories_in_tree(tree, create_mock_data_iter(i), step=i)

    def assert_len(x):
        assert len(x) == L

    jax.tree_map(assert_len, tree)

    # check compound master type
    np.testing.assert_allclose(np.array(tree["compound"]), np.arange(10) * 10)
