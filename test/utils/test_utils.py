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

    assert wa != wb

    assert_equal(wa.wrapped, np.asarray(wa))
    assert wa.wrapped is not wa


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
