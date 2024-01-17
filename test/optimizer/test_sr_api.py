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


@pytest.mark.parametrize(
    "qgt", [nk.optimizer.qgt.QGTJacobianDense, nk.optimizer.qgt.QGTJacobianPyTree]
)
def test_qgt_partial_jacobian_sanitise(qgt):
    with pytest.raises(ValueError):
        qgt(mode="real", holomorphic=True)

    with pytest.raises(ValueError):
        qgt(diag_scale=0.02, rescale_shift=True)


@pytest.mark.parametrize(
    "qgt", [nk.optimizer.qgt.QGTJacobianDense, nk.optimizer.qgt.QGTJacobianPyTree]
)
@pytest.mark.parametrize(
    "args",
    (
        {
            "mode": "real",
            "diag_shift": 0.03,
            "diag_scale": 0.02,
            "chunk_size": 16,
        },
        {
            "holomorphic": True,
            "diag_scale": 0.03,
        },
    ),
)
def test_qgt_partial_jacobian(qgt, args):
    QGT = qgt(**args)
    assert isinstance(QGT, Callable)
    for k, v in args.items():
        assert k in QGT.keywords
        assert QGT.keywords[k] == v


def test_qgt_partial_onthefly():
    args = {"diag_shift": 0.03, "chunk_size": 16}
    QGT = nk.optimizer.qgt.QGTOnTheFly(**args)
    assert isinstance(QGT, Callable)
    for k, v in args.items():
        assert k in QGT.keywords
        assert QGT.keywords[k] == v


@pytest.mark.parametrize(
    "diag_shift", [0.01, lambda _: 0.01, lambda x: 1 / x, "pytree"]
)
def test_diag_shift_schedule(diag_shift):
    # construct a vstate
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        nk.models.RBM(alpha=1),
    )
    vstate.init_parameters()
    vstate.sample()

    if isinstance(diag_shift, str):
        sr = nk.optimizer.SR(diag_shift=vstate.parameters)
        expected_diag_shift = lambda _: vstate.parameters
    else:
        sr = nk.optimizer.SR(diag_shift=diag_shift)
        if isinstance(diag_shift, Callable):
            expected_diag_shift = diag_shift
        else:
            expected_diag_shift = lambda _: diag_shift

    for step_value in [10, 20.0]:
        # ensure that this call is valid
        sr(vstate, vstate.parameters, step_value)

        # check that the diag_shift passed to the QGT is correct
        qgt = sr.lhs_constructor(vstate, step_value)
        assert qgt.diag_shift == expected_diag_shift(step_value)


def test_schedule_err():
    sr = nk.optimizer.SR(diag_shift=lambda _: 0.01)

    with pytest.raises(TypeError):
        sr(None, None)

    sr = nk.optimizer.SR(diag_scale=lambda _: 0.01)

    with pytest.raises(TypeError):
        sr(None, None)


def test_repr():
    sr = nk.optimizer.SR(diag_shift=lambda _: 0.01, diag_scale=lambda _: 0.01)
    assert "SR" in repr(sr)


def test_qgt_auto_diag_scale_passed():
    # See PR/Issue https://github.com/netket/netket/pull/1692
    # diag_scale was not passed to the concrete type

    # construct a vstate
    N = 5
    hi = nk.hilbert.Spin(1 / 2, N)
    vstate = nk.vqs.MCState(
        nk.sampler.MetropolisLocal(hi),
        nk.models.RBM(alpha=1),
    )
    vstate.init_parameters()
    vstate.sample()

    qgt_constructor = nk.optimizer.qgt.QGTAuto(diag_scale=0.2)
    # NOTE: This assumes that the qgt built is a jacobian
    # in the future this might change...
    qgt = qgt_constructor(vstate)
    assert qgt.scale is not None
