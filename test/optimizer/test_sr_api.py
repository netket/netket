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

from typing import Callable

import netket as nk

from .. import common

pytestmark = common.skipif_mpi


def test_deprecated_name():
    with pytest.warns(FutureWarning):
        nk.optim.sr

    with pytest.raises(AttributeError):
        nk.optim.accabalubba

    assert dir(nk.optim) == dir(nk.optimizer)


def test_deprecated_sr():
    with pytest.warns(FutureWarning):
        nk.optimizer.sr.SRLazyCG()

    with pytest.warns(FutureWarning):
        nk.optimizer.sr.SRLazyGMRES()


@pytest.mark.parametrize(
    "qgt", [nk.optimizer.qgt.QGTJacobianDense, nk.optimizer.qgt.QGTJacobianPyTree]
)
def test_qgt_partial_jacobian(qgt):
    # mode and holomorphic can't be both specified for an actual QGT
    # but we'll never create one
    args = {
        "mode": "real",
        "holomorphic": False,
        "diag_shift": 0.03,
        "rescale_shift": True,
        "chunk_size": 16,
    }
    QGT = qgt(**args)
    assert isinstance(QGT, Callable)
    assert QGT.keywords == args


def test_qgt_partial_onthefly():
    args = {"diag_shift": 0.03, "chunk_size": 16}
    QGT = nk.optimizer.qgt.QGTOnTheFly(**args)
    assert isinstance(QGT, Callable)
    assert QGT.keywords == args
