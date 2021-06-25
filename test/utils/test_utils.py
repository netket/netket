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
