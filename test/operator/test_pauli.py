# Copyright 2022 The Netket Authors. - All Rights Reserved.
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
import pytest

import netket as nk


def test_deduced_hilbert_pauli():
    op = nk.operator.PauliStrings(["XXI", "YZX", "IZX"], [0.1, 0.2, -1.4])
    assert op.hilbert.size == 3
    assert len(op.hilbert.local_states) == 2
    assert isinstance(op.hilbert, nk.hilbert.Qubit)
    assert np.allclose(op.hilbert.local_states, (0, 1))


@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(hi, id=str(hi))
        for hi in (nk.hilbert.Spin(1 / 2, 2), nk.hilbert.Qubit(2), None)
    ],
)
def test_pauli(hilbert):
    operators = ["XX", "YZ", "IZ"]
    weights = [0.1, 0.2, -1.4]

    if hilbert is None:
        op = nk.operator.PauliStrings(operators, weights)
    else:
        op = nk.operator.PauliStrings(hilbert, operators, weights)
        assert op.hilbert == hilbert

    op_l = (
        0.1
        * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmax(op.hilbert, 1)
    )
    op_l += (
        0.2
        * nk.operator.spin.sigmay(op.hilbert, 0)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
    )
    op_l -= 1.4 * nk.operator.spin.sigmaz(op.hilbert, 1)

    assert np.allclose(op.to_dense(), op_l.to_dense())

    assert op.to_sparse().shape == op_l.to_sparse().shape


def test_pauli_subtraction():
    op = nk.operator.PauliStrings("XY")
    np.testing.assert_allclose(-op.to_dense(), (-op).to_dense())

    op = nk.operator.PauliStrings("XI") - nk.operator.PauliStrings("IZ")
    op2 = nk.operator.PauliStrings(["XI", "IZ"], [1.0, -1.0])
    np.testing.assert_allclose(op.to_dense(), op2.to_dense())


def test_pauli_simple_constructor():
    operator = "XX"
    weight = 0.3

    op1 = nk.operator.PauliStrings(operator, weight)
    op2 = nk.operator.PauliStrings([operator], [weight])

    assert np.allclose(op1.to_dense(), op2.to_dense())


def test_pauli_simple_constructor_2():
    operators = ["XX", "YZ", "IZ"]
    weight = 0.3

    op1 = nk.operator.PauliStrings(operators, weight)
    op2 = nk.operator.PauliStrings(operators, [weight for _ in operators])

    assert np.allclose(op1.to_dense(), op2.to_dense())


def test_pauli_trivials():
    operators = ["XX", "YZ", "IZ"]
    weights = [0.1, 0.2, -1.4]

    # without weight
    nk.operator.PauliStrings(operators)
    nk.operator.PauliStrings(nk.hilbert.Qubit(2), operators)
    nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), operators)

    # using keywords
    nk.operator.PauliStrings(operators, weights)
    nk.operator.PauliStrings(nk.hilbert.Qubit(2), operators, weights)
    nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), operators, weights)

    nk.operator.PauliStrings.identity(nk.hilbert.Qubit(2))
    nk.operator.PauliStrings.identity(nk.hilbert.Spin(1 / 2, 2))


def test_pauli_cutoff():
    weights = [1, -1, 1]
    operators = ["ZI", "IZ", "XX"]
    op = nk.operator.PauliStrings(operators, weights, cutoff=1e-8)
    hilbert = op.hilbert
    x = np.ones((2,)) * hilbert.local_states[0]
    xp, mels = op.get_conn(x)
    assert xp.shape[-1] == hilbert.size
    assert xp.shape[-2] == 1


def test_pauli_order():
    """Check related to PR #836"""
    coeff1 = 1 + 0.9j
    coeff2 = 0.3 + 0.43j
    op = nk.operator.PauliStrings(["IZXY", "ZZYX"], [coeff1, coeff2])
    op1 = nk.operator.PauliStrings(["IZXY"], [coeff1])
    op2 = nk.operator.PauliStrings(["ZZYX"], [coeff2])
    op1_true = (
        coeff1
        * nk.operator.spin.sigmaz(op.hilbert, 1, dtype=complex)
        * nk.operator.spin.sigmax(op.hilbert, 2)
        * nk.operator.spin.sigmay(op.hilbert, 3)
    )
    op2_true = (
        coeff2
        * nk.operator.spin.sigmaz(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
        * nk.operator.spin.sigmay(op.hilbert, 2)
        * nk.operator.spin.sigmax(op.hilbert, 3)
    )
    assert np.allclose(op1.to_dense(), op1_true.to_dense())
    assert np.allclose(op2.to_dense(), op2_true.to_dense())
    assert np.allclose(op.to_dense(), (op1_true.to_dense() + op2_true.to_dense()))

    v = op.hilbert.all_states()
    vp, mels = op.get_conn_padded(v)
    assert vp.shape[1] == 1
    assert mels.shape[1] == 1


def test_pauli_matmul():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_true_mm = nk.operator.PauliStrings(["Z", "Y"], [1j, -1j])
    op_mm = op1 @ op2
    assert np.allclose(op_mm.to_dense(), op_true_mm.to_dense())

    # more extensive test
    operators1, weights1 = ["XII", "IXY"], [1, 3]
    op1 = nk.operator.PauliStrings(operators1, weights1)
    operators2, weights2 = ["XZZ", "YIZ", "ZII", "IIY"], [1, 0.2, 0.3, 3.1]
    op2 = nk.operator.PauliStrings(operators2, weights2)
    op = op1 @ op2
    op1_true = weights1[0] * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
    op1_true += (
        weights1[1]
        * nk.operator.spin.sigmax(op.hilbert, 1, dtype=complex)
        * nk.operator.spin.sigmay(op.hilbert, 2)
    )
    op2_true = (
        weights2[0]
        * nk.operator.spin.sigmax(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 1)
        * nk.operator.spin.sigmaz(op.hilbert, 2)
    )
    op2_true += (
        weights2[1]
        * nk.operator.spin.sigmay(op.hilbert, 0, dtype=complex)
        * nk.operator.spin.sigmaz(op.hilbert, 2)
    )
    op2_true += weights2[2] * nk.operator.spin.sigmaz(op.hilbert, 0, dtype=complex)
    op2_true += weights2[3] * nk.operator.spin.sigmay(op.hilbert, 2, dtype=complex)
    assert np.allclose((op1_true @ op2_true).to_dense(), op.to_dense())


def test_pauli_add_and_multiply():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1])
    op_true_add = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_add = op1 + op2
    assert np.allclose(op_add.to_dense(), op_true_add.to_dense())
    op_true_multiply = nk.operator.PauliStrings(["X", "Y", "Z"], [-2, 2, 2])
    op_multiply = op2 * 2  # right
    assert np.allclose(op_multiply.to_dense(), op_true_multiply.to_dense())
    op_multiply = 2 * op2  # left
    assert np.allclose(op_multiply.to_dense(), op_true_multiply.to_dense())

    op_add_cte = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1]) + 2
    op_true_add_cte = nk.operator.PauliStrings(["X", "Y", "Z", "I"], [-1, 1, 1, 2])
    assert np.allclose(op_add_cte.to_dense(), op_true_add_cte.to_dense())


@pytest.mark.parametrize(
    "hilbert",
    [
        pytest.param(hi, id=str(hi))
        for hi in (nk.hilbert.Spin(1 / 2, 2), nk.hilbert.Qubit(2), None)
    ],
)
def test_pauli_output(hilbert):
    ha = nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 2), ["IZ", "ZI"], [1.0, 1.0])
    all_states = ha.hilbert.all_states()
    xp, _ = ha.get_conn_padded(all_states)
    xp = xp.reshape(-1, ha.hilbert.size)

    # following will throw an error if the output is not a valid hilbert state
    for xpi in xp:
        assert np.any(xpi == all_states), "{} not in hilbert space {}".format(
            xpi, ha.hilbert
        )


def test_pauli_dense():
    for op in ("I", "X", "Y", "Z"):
        ha1 = nk.operator.PauliStrings(nk.hilbert.Qubit(1), [op], [1])
        ha2 = nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 1), [op], [1])
        assert np.allclose(ha1.to_dense(), ha2.to_dense())


def test_pauli_zero():
    op1 = nk.operator.PauliStrings(["IZ"], [1])
    op2 = nk.operator.PauliStrings(["IZ"], [-1])
    op = op1 + op2

    all_states = op.hilbert.all_states()
    _, mels = op.get_conn_padded(all_states)
    assert np.allclose(mels, 0)


def test_openfermion_conversion():
    # skip test if openfermion not installed
    pytest.importorskip("openfermion")
    from openfermion.ops import QubitOperator

    # first term is a constant
    of_qubit_operator = (
        QubitOperator("") + 0.5 * QubitOperator("X0 X3") + 0.3 * QubitOperator("Z0")
    )

    # no extra info given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator)
    assert isinstance(ps, nk.operator.PauliStrings)
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 4

    # number of qubits given
    ps = nk.operator.PauliStrings.from_openfermion(of_qubit_operator, n_qubits=6)
    assert isinstance(ps, nk.operator.PauliStrings)
    # check default
    assert isinstance(ps.hilbert, nk.hilbert.Qubit)
    assert ps.hilbert.size == 6

    # with hilbert
    hilbert = nk.hilbert.Spin(1 / 2, 6)
    ps = nk.operator.PauliStrings.from_openfermion(hilbert, of_qubit_operator)
    assert ps.hilbert == hilbert
    assert ps.hilbert.size == 6
