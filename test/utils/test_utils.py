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

import pytest
import numpy as np

import jax
import jax.numpy as jnp

from numpy.testing import assert_equal

from netket.jax import PRNGKey, PRNGSeq
from netket.utils import HashableArray
from netket.utils.summation import KahanSum

from .. import common

pytestmark = common.skipif_mpi


def test_PRNGSeq():
    k = PRNGKey(44)
    seq = PRNGSeq()
    k1 = next(seq)
    k2 = next(seq)

    assert k is not k1 and k1 is not k2

    keys = seq.take(4)
    assert keys.shape == (4, 2)

    seq1 = PRNGSeq(12)
    seq2 = PRNGSeq(12)
    assert jnp.all(seq1.take(10) == seq2.take(10))


@pytest.mark.parametrize("numpy", [np, jnp])
def test_HashableArray(numpy):
    a = numpy.asarray(np.random.rand(256, 128))
    b = 2 * a

    wa = HashableArray(a)
    wa2 = HashableArray(a.copy())
    wb = HashableArray(b)

    assert hash(wa) == hash(wa2)
    assert wa == wa2

    assert hash(wb) == hash(wb)
    assert wb == wb

    assert wa != a
    assert wa != wb

    # check __eq__ does not fail on two arrays w/ different shape
    wc = HashableArray(a[np.newaxis])
    assert wc != wa
    # check __eq__ does not fail on two arrays w/ different dtype
    wd = HashableArray(a.astype(int))
    assert wd != wa

    assert_equal(wa.wrapped, np.asarray(wa))
    assert_equal(wa.wrapped, jnp.asarray(wa))
    assert wa.wrapped is not wa

    # test construction from hashable array
    wa2 = HashableArray(a)
    assert hash(wa) == hash(wa2)
    assert wa == wa2
    assert_equal(wa2.wrapped, np.asarray(wa))
    assert_equal(wa2.wrapped, jnp.asarray(wa))

    # test construction from jax array
    wa3 = HashableArray(jnp.array(a))
    assert hash(wa) == hash(wa3)
    assert wa == wa3
    assert_equal(wa3.wrapped, np.asarray(wa))
    assert_equal(wa3.wrapped, jnp.asarray(wa))

    # Check that it is a leaf object, and not a pytree
    leafs, _ = jax.tree_util.tree_flatten(wa)
    assert len(leafs) == 1
    assert leafs[0] == wa

    # ensure that repr and str work
    assert isinstance(repr(wa), str)
    assert isinstance(str(wa), str)


def test_Kahan_sum():
    ksum1 = KahanSum(0.0)
    ksum2 = KahanSum(0.0)
    vals = 0.01 * np.ones(500)
    for val in vals:
        ksum1 += val
        ksum2 = ksum2 + val

    assert ksum1.value == pytest.approx(5.0, rel=1e-15, abs=1e-15)
    assert ksum2.value == pytest.approx(5.0, rel=1e-15, abs=1e-15)


def test_batching_wrapper():
    from netket.utils import wrap_to_support_scalar

    def applyfun(pars, x, mutable=False):
        # this assert fails if the wrapper is not working
        assert x.ndim > 1
        if not mutable:
            return x.sum(axis=-1)
        else:
            return (x.sum(axis=-1), {})

    # check same hash
    assert hash(wrap_to_support_scalar(applyfun)) == hash(
        wrap_to_support_scalar(applyfun)
    )

    afun = wrap_to_support_scalar(applyfun)

    x = jnp.ones(5)
    xb = jnp.ones((1, 5))

    # no mutable state
    res = afun(None, x)
    assert res.shape == ()
    assert res == jnp.sum(x, axis=-1)
    res = afun(None, xb)
    assert res.shape == (1,)
    assert res == jnp.sum(x, axis=-1)

    # mutable state
    res = afun(None, x, mutable=True)[0]
    assert res.shape == ()
    assert res == jnp.sum(x, axis=-1)
    res = afun(None, xb, mutable=True)[0]
    assert res.shape == (1,)
    assert res == jnp.sum(x, axis=-1)


def test_deprecated_dispatch_bool():
    from netket.utils import dispatch

    @dispatch.dispatch
    def test(a):
        return 1

    with pytest.warns():

        @dispatch.dispatch
        def test(a: dispatch.TrueT):  # noqa: F811
            return True

    with pytest.warns():

        @dispatch.dispatch
        def test(b: dispatch.FalseT):  # noqa: F811
            return False

    assert test(1) == 1
    assert test(True) is True
    assert test(False) is False

    with pytest.warns():

        @dispatch.dispatch
        def test(b: dispatch.Bool):  # noqa: F811
            return False
