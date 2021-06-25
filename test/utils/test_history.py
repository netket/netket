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

import netket as nk

import numpy as np
import jax
import jax.numpy as jnp
from dataclasses import dataclass

from .. import common

pytestmark = common.skipif_mpi


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
        "jaxcomplex": jnp.array(iter + 1j * iter),
        "dict": {"int": iter},
        "compound": MockCompoundType(iter, iter * 10),
        "mockdict": MockDictType(iter, iter * 10),
        "mock": MockClass(iter),
        "matrix": np.full((3, 4), iter),
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
