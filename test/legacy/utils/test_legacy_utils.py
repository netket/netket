# Copyright 2020 The Simons Foundation, Inc. - All Rights Reserved.
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

from netket.legacy.vmc_common import tree_map
import pytest

pytestmark = pytest.mark.legacy


def test_tree_map():
    # Structure: name -> (input, expected)
    test_trees = {
        "None": (None, None),
        "leaf": (0, 1),
        "list": ([0, 1], [1, 2]),
        "tuple1": ((0,), (1,)),
        "tuple3": ((0, 1, 0), (1, 2, 1)),
        "list of list": ([[0, 1], [1], [0, 1, 0]], [[1, 2], [2], [1, 2, 1]]),
        "dict": ({"a": 0, "b": 1}, {"a": 1, "b": 2}),
        "mixed list": ([(0, 1), [1], 1, {"a": 0}], [(1, 2), [2], 2, {"a": 1}]),
    }

    for name, (inp, expected) in test_trees.items():
        info = "{}, input={}".format(name, inp)
        assert tree_map(lambda x: x + 1, inp) == expected, info
