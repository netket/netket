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

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

import flax

import netket as nk

from .. import common

pytestmark = common.skipif_distributed


@dataclass
class MockCompoundType:
    field1: int
    field2: float

    def to_compound(self):
        return "field2", {"field1": self.field1, "field2": self.field2}


@dataclass
class MockDictType:
    field1: int
    field2: float

    def to_dict(self):
        return {"field1": self.field1, "field2": self.field2}


@dataclass
class MockClass:
    field1: int


def create_mock_data_iter(iter):
    return {
        "int": iter,
        "complex": iter + 1j * iter,
        "npint": np.array(iter),
        "jax-1d": jnp.full((1,), iter),  # tests logging a 1D, 1 element vector
        "jaxcomplex": jnp.array(iter + 1j * iter),
        "jaxcomple-1d": jnp.full((1,), iter + 1j * iter),
        "dict": {"int": iter},
        "frozendict": flax.core.freeze({"sub": {"int": iter}}),
        "compound": MockCompoundType(iter, iter * 10),
        "mockdict": MockDictType(iter, iter * 10),
        "mock": MockClass(iter),
        "matrix": np.full((3, 4), iter),
        "empty": {},
    }


def test_accum_mvhistory():
    L = 10

    tree = None
    for i in range(L):
        tree = nk.utils.accum_histories_in_tree(tree, create_mock_data_iter(i), step=i)

    def assert_len(x):
        assert len(x) == L

    jax.tree_util.tree_map(assert_len, tree)

    # check compound master type
    np.testing.assert_allclose(np.array(tree["compound"]), np.arange(10) * 10)

    # test that repr does not fail
    repr(tree)

    # check frozen
    np.testing.assert_allclose(
        np.array(tree["frozendict"]["sub"]["int"]), np.arange(10)
    )

    # Check that empty is not in the accumulated tree
    assert "empty" not in tree


def test_append():
    a1 = nk.utils.History(create_mock_data_iter(0))
    a2 = nk.utils.History(create_mock_data_iter(1), iters=1)
    a1.append(a2)

    assert set(a1.keys()) == set(a2.keys())
    for key in a1.keys():
        assert len(a1[key]) == 2
    assert all(a1.iters == np.arange(2))

    a0 = a1[-1]
    for key in a1.keys():
        assert len(a0[key]) == 1
    len(a1.iters) == 1
    a1.iters[0] == 1

    a0 = a1[0:]
    for key in a1.keys():
        assert len(a0[key]) == 2
    assert all(a1.iters == np.arange(2))

    # test that repr does not fail
    repr(a1)
    repr(a2)


def test_construct_from_dict():
    tree = nk.utils.History(create_mock_data_iter(0))
    a2 = nk.utils.History(create_mock_data_iter(1), iters=1)
    tree.append(a2)

    new_tree = nk.utils.History(tree.to_dict())
    assert len(new_tree) == len(tree)
    assert set(new_tree.keys()) == set(tree.keys())
    np.testing.assert_allclose(new_tree.iters, tree.iters)
    for key in tree.keys():
        np.testing.assert_equal(new_tree[key], tree[key])

    tree.append(new_tree)
    assert len(tree.iters) == len(new_tree.iters) * 2
    for key in tree.keys():
        len(tree[key]) == len(new_tree[key]) * 2


def test_historydict():
    L = 10

    tree = nk.utils.history.HistoryDict()
    for i in range(L):
        tree = nk.utils.accum_histories_in_tree(tree, create_mock_data_iter(i), step=i)

    # There are no HistoryDict inside
    tree_dict = tree.to_dict()
    leafs = jax.tree.leaves(tree_dict)
    assert all(not isinstance(l, nk.utils.history.HistoryDict) for l in leafs)

    # But they are created on the fly
    assert isinstance(tree["dict"], nk.utils.history.HistoryDict)

    assert isinstance(repr(tree), str)

    # Even if you put it in, it gets out
    new_tree = nk.utils.history.HistoryDict({"histdict": tree["dict"]})
    assert all(
        not isinstance(l, nk.utils.history.HistoryDict)
        for l in jax.tree.leaves(new_tree.to_dict())
    )
