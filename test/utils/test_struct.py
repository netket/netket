# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for flax.struct."""

from typing import Any
import pytest
from functools import partial

import dataclasses

from netket.utils import struct

import jax

from .. import common

pytestmark = common.skipif_mpi


@struct.dataclass
class Point0:
    x: float
    y: float
    meta: Any = struct.field(pytree_node=False)

    def __pre_init__(self, *args, **kwargs):
        if "z" in kwargs:
            kwargs["x"] = kwargs.pop("z") * 10

        return args, kwargs

    @struct.property_cached
    def cached_node(self) -> int:
        return 3


@struct.dataclass
class Point1:
    x: float
    y: float
    meta: Any = struct.field(pytree_node=False)

    @struct.property_cached
    def cached_node(self) -> int:
        return 3


@struct.dataclass
class Point1Child(Point1):
    z: float

    @struct.property_cached
    def cached_node(self) -> int:
        return 4


@struct.dataclass
class Point1Child2(Point1):
    z: float


Point1ChildConstructor = partial(Point1Child, z=3)
Point1Child2Constructor = partial(Point1Child2, z=3)


@pytest.mark.parametrize(
    "PointT", [Point0, Point1, Point1ChildConstructor, Point1Child2Constructor]
)
def test_no_extra_fields(PointT):
    p = PointT(x=1, y=2, meta={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.new_field = 1


@pytest.mark.parametrize(
    "PointT", [Point0, Point1, Point1ChildConstructor, Point1Child2Constructor]
)
def test_mutation(PointT):
    p = PointT(x=1, y=2, meta={})
    new_p = p.replace(x=3)
    assert new_p == PointT(x=3, y=2, meta={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.y = 3


@pytest.mark.parametrize("PointT", [Point0, Point1])
def test_pytree_nodes(PointT):
    p = PointT(x=1, y=2, meta={"abc": True})
    leaves = jax.tree_util.tree_leaves(p)
    assert leaves == [1, 2]
    new_p = jax.tree_map(lambda x: x + x, p)
    assert new_p == PointT(x=2, y=4, meta={"abc": True})


def test_pytree_nodes_inheritance():
    p = Point1Child(x=1, y=2, z=3, meta={"abc": True})
    _ = Point1Child(1, 2, {"abc": True}, 3)
    leaves = jax.tree_util.tree_leaves(p)
    assert leaves == [1, 2, 3]
    new_p = jax.tree_map(lambda x: x + x, p)
    assert new_p == Point1Child(x=2, y=4, z=6, meta={"abc": True})


@pytest.mark.parametrize("PointT", [Point0, Point1, Point1Child2Constructor])
def test_cached_property(PointT):
    p = PointT(x=1, y=2, meta={"abc": True})

    assert p.__cached_node_cache is struct.Uninitialized
    assert p.cached_node == 3
    assert p.__cached_node_cache == 3

    p = p.replace(x=1)
    assert p.__cached_node_cache is struct.Uninitialized
    p._precompute_cached_properties()
    assert p.__cached_node_cache == 3


def test_cached_property_inheritance():
    p = Point1Child(x=1, y=2, z=3, meta={"abc": True})

    assert p.__cached_node_cache is struct.Uninitialized
    assert p.cached_node == 4
    assert p.__cached_node_cache == 4

    p = p.replace(x=1)
    assert p.__cached_node_cache is struct.Uninitialized
    p._precompute_cached_properties()
    assert p.__cached_node_cache == 4


def test_pre_init_property():
    p = Point0(z=1, y=2, meta={"abc": True})

    assert p.x == 10


def test_inheritance():
    p = Point1Child(x=1, z=1, y=2, meta={"abc": True})

    assert p.x == 1
    assert p.z == 1


@struct.dataclass(cache_hash=True)
class Point0cache:
    x: float

    def __hash__(self):
        return 123


def test_cache_hash():
    a = Point0cache(1)
    assert a.__Point0cache_hash_cache is struct.Uninitialized
    hash(a) == 123
    assert a.__Point0cache_hash_cache == 123
    object.__setattr__(a, "__Point0cache_hash_cache", 1234)
    hash(a) == 1234


@struct.dataclass
class PointC:
    x: float
    y: float

    @struct.property_cached(pytree_node=True)
    def cached_node(self) -> int:
        return self.x * self.y


# test for pytree_node=True cached properties
def test_cached_pytreenode_properties():
    # check cache uninitialized then populated
    p = PointC(2.0, 3.0)
    assert p.__cached_node_cache is struct.Uninitialized
    assert p.cached_node == 2.0 * 3.0
    assert p.__cached_node_cache == 6.0

    # check cache reset after call to replace
    p = p.replace(y=3.0)
    assert p.__cached_node_cache is struct.Uninitialized
    assert p.cached_node == 2.0 * 3.0
    assert p.__cached_node_cache == 6.0

    @jax.jit
    def compute(x: PointC):
        return x.cached_node * 3, x

    # check cache populated when returned from jitted function
    p = PointC(2.0, 3.0)
    res, p2 = compute(p)
    assert p.__cached_node_cache is struct.Uninitialized
    assert res == 6.0 * 3
    assert p2.__cached_node_cache == 6.0

    # check cache not modified by jit
    res, p3 = compute(p2)
    assert res == 6.0 * 3
    assert p3.__cached_node_cache == 6.0
