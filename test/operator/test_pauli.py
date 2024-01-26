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

import pytest

import numpy as np
import jax
import jax.numpy as jnp

import netket as nk
from netket.utils import array_in


def test_deduced_hilbert_pauli():
    op = nk.operator.PauliStrings(["XXI", "YZX", "IZX"], [0.1, 0.2, -1.4])
    assert op.hilbert.size == 3
    assert len(op.hilbert.local_states) == 2
    assert isinstance(op.hilbert, nk.hilbert.Qubit)
    np.testing.assert_allclose(op.hilbert.local_states, (0, 1))


def test_pauli_tensorhilbert():
    hi = nk.hilbert.Spin(0.5, 2, total_sz=0) * nk.hilbert.Spin(0.5, 1)
    op = nk.operator.PauliStrings(hi, ["XXI", "YYY", "IZX"], [0.1, 0.2, -1.4])
    assert op.hilbert.size == 3
    s = hi.all_states()
    sp, _ = op.get_conn_padded(s)
    sp = sp.reshape(-1, 3)
    for _s in sp:
        assert array_in(_s, s)


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

    np.testing.assert_allclose(op.to_dense(), op_l.to_dense())

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

    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())


def test_pauli_simple_constructor_2():
    operators = ["XX", "YZ", "IZ"]
    weight = 0.3

    op1 = nk.operator.PauliStrings(operators, weight)
    op2 = nk.operator.PauliStrings(operators, [weight for _ in operators])

    np.testing.assert_allclose(op1.to_dense(), op2.to_dense())


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
    np.testing.assert_allclose(op1.to_dense(), op1_true.to_dense())
    np.testing.assert_allclose(op2.to_dense(), op2_true.to_dense())
    np.testing.assert_allclose(
        op.to_dense(), (op1_true.to_dense() + op2_true.to_dense())
    )

    v = op.hilbert.all_states()
    vp, mels = op.get_conn_padded(v)
    assert vp.shape[1] == 1
    assert mels.shape[1] == 1


def test_pauli_matmul():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_true_mm = nk.operator.PauliStrings(["Z", "Y"], [1j, -1j])
    op_mm = op1 @ op2
    np.testing.assert_allclose(op_mm.to_dense(), op_true_mm.to_dense())

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
    np.testing.assert_allclose((op1_true @ op2_true).to_dense(), op.to_dense())


def test_pauli_add_and_multiply():
    op1 = nk.operator.PauliStrings(["X"], [1])
    op2 = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1])
    op_true_add = nk.operator.PauliStrings(["Y", "Z"], [1, 1])
    op_add = op1 + op2
    np.testing.assert_allclose(op_add.to_dense(), op_true_add.to_dense())
    op_true_multiply = nk.operator.PauliStrings(["X", "Y", "Z"], [-2, 2, 2])
    op_multiply = op2 * 2  # right
    np.testing.assert_allclose(op_multiply.to_dense(), op_true_multiply.to_dense())
    op_multiply = 2 * op2  # left
    np.testing.assert_allclose(op_multiply.to_dense(), op_true_multiply.to_dense())

    op_add_cte = nk.operator.PauliStrings(["X", "Y", "Z"], [-1, 1, 1]) + 2
    op_true_add_cte = nk.operator.PauliStrings(["X", "Y", "Z", "I"], [-1, 1, 1, 2])
    np.testing.assert_allclose(op_add_cte.to_dense(), op_true_add_cte.to_dense())

    # test multiplication and addition with numpy/jax scalar
    op1 = np.array(0.5) * nk.operator.PauliStrings(["X"], [1])
    op1_true = nk.operator.PauliStrings(["X"], [0.5])
    np.testing.assert_allclose(op1.to_dense(), op1_true.to_dense())

    op1 = jnp.array(0.5) * nk.operator.PauliStrings(["X"], [1])
    np.testing.assert_allclose(op1.to_dense(), op1_true.to_dense())

    op1 = np.array(0.5) + nk.operator.PauliStrings(["X"], [1])
    op1_true = nk.operator.PauliStrings(["I", "X"], [0.5, 1])
    np.testing.assert_allclose(op1.to_dense(), op1_true.to_dense())

    op1 = jnp.array(0.5) + nk.operator.PauliStrings(["X"], [1])
    np.testing.assert_allclose(op1.to_dense(), op1_true.to_dense())


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
        assert np.any(xpi == all_states), "{xpi} not in hilbert space {ha.hilbert}"


def test_pauli_dense():
    for op in ("I", "X", "Y", "Z"):
        ha1 = nk.operator.PauliStrings(nk.hilbert.Qubit(1), [op], [1])
        ha2 = nk.operator.PauliStrings(nk.hilbert.Spin(1 / 2, 1), [op], [1])
        np.testing.assert_allclose(ha1.to_dense(), ha2.to_dense())


def test_pauli_zero():
    op1 = nk.operator.PauliStrings(["IZ"], [1])
    op2 = nk.operator.PauliStrings(["IZ"], [-1])
    op = op1 + op2

    all_states = op.hilbert.all_states()
    _, mels = op.get_conn_padded(all_states)
    np.testing.assert_allclose(mels, 0)


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


def test_pauli_jax_sparse_works():
    hi = nk.hilbert.Spin(0.5, 9)
    g = nk.graph.Square(3)
    ham = nk.operator.Ising(hi, g, h=1.0)
    ham_jax_sp = ham.to_local_operator().to_pauli_strings().to_jax_operator()
    ham_jax_d = ham_jax_sp.to_sparse().todense()

    ham_d = ham.to_dense()
    np.testing.assert_allclose(ham_jax_d, ham_d)


def test_pauli_problem():
    x1 = nk.operator.PauliStringsJax("XII")
    x2 = nk.operator.PauliStringsJax("IXI")
    x3 = x1 @ x2
    dtype = jax.dtypes.canonicalize_dtype(float)
    assert x1.weights.dtype == dtype
    assert x2.weights.dtype == dtype
    assert x3.weights.dtype == dtype

    assert (x1 + x1 @ x2).weights.dtype == dtype


def test_pauliY_dtype():
    ham = nk.operator.PauliStrings("XXX", dtype=np.float32)
    assert ham.dtype == np.float32
    ham = nk.operator.PauliStrings("XYY", dtype=np.float32)
    assert ham.dtype == np.float32
    ham = nk.operator.PauliStrings(["XXX", "XYY"], dtype=np.float32)
    assert ham.dtype == np.float32

    ham = nk.operator.PauliStrings("XXY", dtype=np.complex64)
    assert ham.dtype == np.complex64
    ham = nk.operator.PauliStrings(["XXX", "XXY"], dtype=np.complex64)
    assert ham.dtype == np.complex64

    ham = nk.operator.PauliStrings("XXX", [1j], dtype=np.complex64)
    assert ham.dtype == np.complex64
    ham = nk.operator.PauliStrings("XYY", [1j], dtype=np.complex64)
    assert ham.dtype == np.complex64
    ham = nk.operator.PauliStrings(["XXX", "XYY"], [1, 1j], dtype=np.complex64)
    assert ham.dtype == np.complex64

    with pytest.raises(TypeError):
        ham = nk.operator.PauliStrings("XXY", dtype=np.float32)
    with pytest.raises(TypeError):
        ham = nk.operator.PauliStrings(["XXX", "XXY"], dtype=np.float32)

    with pytest.raises(TypeError):
        ham = nk.operator.PauliStrings("XXX", [1j], dtype=np.float32)
    with pytest.raises(TypeError):
        ham = nk.operator.PauliStrings("XYY", [1j], dtype=np.float32)
    with pytest.raises(TypeError):
        ham = nk.operator.PauliStrings(["XXX", "XYY"], [1, 1j], dtype=np.float32)


def test_pauli_empty_constructor_error():
    with pytest.raises(ValueError, match=r".*the hilbert space must be specified.*"):
        nk.operator.PauliStrings([])


operators_to_test = [
    pytest.param(nk.operator.PauliStrings("X", dtype=np.float32), id="X"),
    pytest.param(nk.operator.PauliStrings("Z", dtype=np.complex64), id="Z_complex"),
    pytest.param(nk.operator.PauliStrings("Y", dtype=np.complex64), id="Y"),
    pytest.param(nk.operator.PauliStrings("X", [1j], dtype=np.complex64), id="X_1j"),
]


@pytest.mark.parametrize("b", operators_to_test)
@pytest.mark.parametrize("a", operators_to_test)
def test_pauli_inplace(a, b):
    if a.dtype == np.float32 and b.dtype == np.complex64:
        with pytest.raises(TypeError):
            a += b
    else:
        a1 = a.copy()
        a1 += b
        assert a1.dtype == a.dtype
        np.testing.assert_allclose(a1.to_dense(), (a + b).to_dense())

    # Currently DiscreteOperator does not implement __imatmul__,
    # so imatmul will call __matmul__ and may change dtype
    a1 = a.copy()
    a1 @= b
    np.testing.assert_allclose(a1.to_dense(), (a @ b).to_dense())


@pytest.mark.parametrize("cutoff", [1.0e-3, 0.0, None])
@pytest.mark.parametrize("Op", [nk.operator.PauliStrings, nk.operator.PauliStringsJax])
def test_cutoff(Op, cutoff):
    coeff = 1.0e-5
    hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=4)
    if cutoff is not None:
        ha = Op(hi, ["XXII", "YYII"], [1.0, 1.0 + coeff], cutoff=cutoff)
    else:
        # cutoff=None is not yet part of the public API
        ha = Op(hi, ["XXII", "YYII"], [1.0, 1.0 + coeff], cutoff=0)
        ha._cutoff = None
    # numba n_conn does not support a single sample, so we add a dummy axis here
    x = jnp.array([-1, -1, 1, 1])[None]
    n_conn = ha.n_conn(x)
    xp, mels = ha.get_conn_padded(x)
    np.testing.assert_equal(ha.max_conn_size, 1)
    if cutoff is not None and np.abs(coeff) < cutoff:
        np.testing.assert_array_equal(n_conn, np.array([0]))
        np.testing.assert_array_equal(mels, 0.0)
        np.testing.assert_array_equal(xp, x[None])
    else:
        np.testing.assert_array_equal(n_conn, np.array([1]))
        np.testing.assert_allclose(mels, -coeff)
        np.testing.assert_array_equal(xp, np.array([[[1, 1, 1, 1]]]))


@pytest.mark.parametrize("Op", [nk.operator.PauliStrings, nk.operator.PauliStringsJax])
def test_cutoff_constrained(Op):
    hi = nk.hilbert.Spin(s=1 / 2, total_sz=0, N=4)
    ha = Op(hi, ["XXII", "YYII"], [1, 1], cutoff=0, dtype=int)
    x = jnp.array([-1, -1, 1, 1])
    xp, mels = ha.get_conn_padded(x)
    np.testing.assert_array_equal(mels, 0)
    np.testing.assert_array_equal(xp, x[None])
