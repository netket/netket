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

import warnings

import pytest

from collections.abc import Callable

import netket as nk

from .. import common

pytestmark = common.skipif_distributed


@pytest.mark.parametrize(
    "qgt", [nk.optimizer.qgt.QGTJacobianDense, nk.optimizer.qgt.QGTJacobianPyTree]
)
def test_qgt_partial_jacobian_sanitise(qgt):
    with pytest.raises(ValueError):
        qgt(mode="real", holomorphic=True)


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


@pytest.mark.parametrize(
    "qgt", [nk.optimizer.qgt.QGTJacobianDense, nk.optimizer.qgt.QGTJacobianPyTree]
)
def test_sr_diag_warnings(qgt):
    """
    Test various scenarios for warnings related to diag_shift and diag_scale in SR.
    """
    warning_message = r"Constructing the SR object with `SR\(qgt= MyQGTType\({.*}\)\)` can lead to unexpected results and has been deprecated, because the keyword arguments specified in the QGTType are overwritten by those specified by the SR class and its defaults\.\n\nTo fix this, construct SR as  `SR\(qgt=MyQGTType, {.*}\)` \.\n\nIn the future, this warning will become an error\."

    # Case 1: Overwriting diag_shift from SR default
    with pytest.warns(UserWarning, match=warning_message):
        nk.optimizer.SR(qgt=qgt(diag_shift=1e-3))

    # Case 2: Overwriting diag_scale from SR default
    with pytest.warns(UserWarning, match=warning_message):
        nk.optimizer.SR(qgt=qgt(diag_scale=1e-4))

    # Case 3: Overwriting both diag_shift and diag_scale from SR default
    with pytest.warns(UserWarning, match=warning_message):
        nk.optimizer.SR(qgt=qgt(diag_shift=1e-3, diag_scale=1e-4))

    # Case 4: Warning with default diag_shift and diag_scale by specifying them in SR
    with pytest.warns(UserWarning, match=warning_message):
        nk.optimizer.SR(
            qgt=qgt(diag_shift=1e-3, diag_scale=1e-4), diag_shift=1e-2, diag_scale=1e-3
        )

    # Case 5: No warning when diag_shift and diag_scale are specified only in SR
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        nk.optimizer.SR(qgt=qgt)
    assert len(w) == 0, "Unexpected warning(s) raised"
