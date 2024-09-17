import warnings

import numpy as np
from scipy import sparse

import netket as nk
from netket.operator.boson import (
    create as bcreate,
    destroy as bdestroy,
    number as bnumber,
)
from netket.operator.spin import sigmax, sigmay, sigmaz, sigmam, sigmap
from netket.operator import AbstractOperator, LocalOperator
from netket.utils import module_version

import pytest
from pytest import raises

from .. import common

# TODO: once we require np 2.0.0, we can remove this
if module_version(np) >= (2, 0, 0):
    from numpy.exceptions import ComplexWarning
else:
    from numpy import ComplexWarning

import jax

herm_operators = {}
generic_operators = {}

# Custom Hamiltonian
sx = [[0, 1], [1, 0]]
sy = [[0, -1.0j], [1.0j, 0]]
sz = [[1, 0], [0, -1]]
sm = [[0, 0], [1, 0]]
sp = [[0, 1], [0, 0]]
g = nk.graph.Graph(edges=[[i, i + 1] for i in range(8)])
hi = nk.hilbert.CustomHilbert(local_states=nk.utils.StaticRange(-1, 2, 2), N=g.n_nodes)

sy_sparse = sparse.csr_matrix(sy)


def _loc(*args):
    return nk.operator.LocalOperator(hi, *args)


sx_hat = _loc([sx] * 3, [[0], [1], [5]])
sy_hat = _loc([sy] * 4, [[2], [3], [4], [7]])
szsz_hat = _loc(sz, [0]) @ _loc(sz, [1])
szsz_hat += _loc(sz, [4]) @ _loc(sz, [5])
szsz_hat += _loc(sz, [6]) @ _loc(sz, [8])
szsz_hat += _loc(sz, [7]) @ _loc(sz, [0])
sy_sparse_hat = _loc([sy_sparse] * 3, [[0], [1], [5]])

herm_operators["sx (real op)"] = sx_hat
herm_operators["sy"] = sy_hat
herm_operators["sy_sparse"] = sy_sparse_hat

herm_operators["Custom Hamiltonian"] = sx_hat + sy_hat + szsz_hat
herm_operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + (2.0 * sy_hat)

sm_hat = nk.operator.LocalOperator(hi, [sm] * 3, [[0], [1], [4]])
sp_hat = nk.operator.LocalOperator(hi, [sp] * 3, [[0], [1], [4]])


generic_operators["sigma +/-"] = (sm_hat, sp_hat)


def assert_same_matrices(matl, matr, eps=1.0e-6):
    if isinstance(matl, AbstractOperator):
        matl = matl.to_dense()
    elif sparse.issparse(matl):
        matl = matl.todense()

    if isinstance(matr, AbstractOperator):
        matr = matr.to_dense()
    elif sparse.issparse(matr):
        matr = matr.todense()

    np.testing.assert_allclose(matl, matr, atol=eps, rtol=eps)


@pytest.mark.parametrize(
    "op",
    [pytest.param(op, id=name) for name, op in herm_operators.items()],
)
def test_hermitian_local_operator_transpose_conjugation(op):
    orig_op = op.copy()

    op_t = op.transpose()
    op_c = op.conjugate()
    op_h = op.transpose().conjugate()

    assert [
        assert_same_matrices(m1, m2)
        for (m1, m2) in zip(op._operators, orig_op._operators)
    ]

    mat = op.to_dense()
    mat_t = op_t.to_dense()
    mat_c = op_c.to_dense()
    mat_h = op_h.to_dense()

    assert [
        assert_same_matrices(m1, m2)
        for (m1, m2) in zip(op._operators, orig_op._operators)
    ]

    assert_same_matrices(mat, mat_h)
    assert_same_matrices(mat_t, mat_c)

    mat_t_t = op.transpose().transpose().to_dense()
    mat_c_c = op.conjugate().conjugate().to_dense()

    assert_same_matrices(mat, mat_t_t)
    assert_same_matrices(mat, mat_c_c)

    assert [
        assert_same_matrices(m1, m2)
        for (m1, m2) in zip(op._operators, orig_op._operators)
    ]


@pytest.mark.parametrize(
    "op_tuple",
    [pytest.param(op, id=name) for name, op in generic_operators.items()],
)
def test_local_operator_transpose_conjugation(op_tuple):
    op, oph = op_tuple

    mat = op.to_dense()
    math = oph.to_dense()

    mat_h = op.transpose().conjugate().to_dense()
    assert_same_matrices(mat_h, math)

    math_h = oph.transpose().conjugate().to_dense()
    assert_same_matrices(math_h, mat)


def test_lazy_operator_matdensevec():
    sz0 = nk.operator.spin.sigmaz(hi, 0)
    v_np = np.random.rand(hi.n_states)
    v_jx = jax.numpy.asarray(v_np)

    sz0_t = sz0.transpose()
    assert_same_matrices(sz0_t @ v_np, sz0_t.to_dense() @ v_np)
    assert_same_matrices(sz0_t @ v_jx, sz0_t.to_dense() @ v_jx)

    sz0_h = sz0.transpose().conjugate()
    assert_same_matrices(sz0_h @ v_np, sz0_h.to_dense() @ v_np)
    assert_same_matrices(sz0_h @ v_jx, sz0_h.to_dense() @ v_jx)

    sz0_2 = sz0_h @ sz0
    assert_same_matrices(sz0_2 @ v_np, sz0_2.to_dense() @ v_np)
    assert_same_matrices(sz0_2 @ v_jx, sz0_2.to_dense() @ v_jx)


def test_local_operator_add():
    sz0 = nk.operator.spin.sigmaz(hi, 0)

    ham = 0.5 * sz0.to_dense()
    ha = 0.5 * sz0
    ha2 = nk.operator.spin.sigmaz(hi, 0)
    ha2 *= 0.5
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    ha = ha * 1j
    with raises(ValueError):
        ha2 *= 1j

    ha2 = ha2 * 1j
    ham = ham * 1j
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    for i in range(1, 3):
        ha = ha + 0.2 * nk.operator.spin.sigmaz(hi, i)
        ha2 += 0.2 * nk.operator.spin.sigmaz(hi, i)
        ham += 0.2 * nk.operator.spin.sigmaz(hi, i).to_dense()
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    for i in range(3, 5):
        ha = ha + 0.2 * nk.operator.spin.sigmax(hi, i)
        ha2 += 0.2 * nk.operator.spin.sigmax(hi, i)
        ham += 0.2 * nk.operator.spin.sigmax(hi, i).to_dense()
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    for i in range(5, 7):
        ha = ha - 0.3 * nk.operator.spin.sigmam(hi, i)
        ha2 -= 0.3 * nk.operator.spin.sigmam(hi, i)
        ham -= 0.3 * nk.operator.spin.sigmam(hi, i).to_dense()
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    ha = ha - 0.3j * nk.operator.spin.sigmam(hi, 7)
    ha2 -= 0.3j * nk.operator.spin.sigmam(hi, 7)
    ham -= 0.3j * nk.operator.spin.sigmam(hi, 7).to_dense()
    assert_same_matrices(ha, ha2)
    assert_same_matrices(ha, ham)

    # test commutativity
    ha = LocalOperator(hi)
    ha2 = LocalOperator(hi, dtype=complex)
    for i in range(0, 3):
        ha += 0.3 * nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmax(hi, i + 1)
        ha += 0.4 * nk.operator.spin.sigmaz(hi, i)
        ha2 += 0.5 * nk.operator.spin.sigmay(hi, i)

    ha_ha2 = ha + ha2
    ha2_ha = ha2 + ha
    assert_same_matrices(ha_ha2, ha2_ha)


def test_simple_operators():
    L = 4
    hi = nk.hilbert.Spin(0.5) ** L

    sx = [[0, 1], [1, 0]]
    sy = [[0, -1.0j], [1.0j, 0]]
    sz = [[1, 0], [0, -1]]
    sm = [[0, 0], [1, 0]]
    sp = [[0, 1], [0, 0]]

    print("Testing Sigma_x/y/z...")
    for i in range(L):
        sx_hat = nk.operator.LocalOperator(hi, sx, [i])
        sy_hat = nk.operator.LocalOperator(hi, sy, [i])
        sz_hat = nk.operator.LocalOperator(hi, sz, [i])
        assert_same_matrices(sigmax(hi, i), sx_hat)
        assert_same_matrices(sigmay(hi, i), sy_hat)
        assert_same_matrices(sigmaz(hi, i), sz_hat)

    print("Testing Sigma_+/-...")
    for i in range(L):
        sm_hat = nk.operator.LocalOperator(hi, sm, [i])
        sp_hat = nk.operator.LocalOperator(hi, sp, [i])
        assert_same_matrices(sigmam(hi, i), sm_hat)
        assert_same_matrices(sigmap(hi, i), sp_hat)

    print("Testing Sigma_+/- composition...")

    hi = nk.hilbert.Spin(0.5, N=L)
    for i in range(L):
        sx = sigmax(hi, i)
        sy = sigmay(hi, i)
        sigmam_hat = 0.5 * (sx + (-1j) * sy)
        sigmap_hat = 0.5 * (sx + (1j) * sy)
        assert_same_matrices(sigmam(hi, i), sigmam_hat)
        assert_same_matrices(sigmap(hi, i), sigmap_hat)

    print("Testing create/destroy composition...")
    hi = nk.hilbert.Fock(3, N=L)
    for i in range(L):
        print("i=", i)
        a = bdestroy(hi, i)
        ad = bcreate(hi, i)
        n = bnumber(hi, i)

        assert_same_matrices(n, ad @ a)
        assert_same_matrices(ad, a.conjugate().transpose())

    print("Testing mixed spaces...")
    L = 3
    his = nk.hilbert.Spin(0.5, N=L)
    hib = nk.hilbert.Fock(3, N=L - 1)
    hi = his * hib
    for i in range(hi.size):
        print("i=", i)
        sx = sigmax(hi, i)

        assert sx.operators[0].shape == (hi.shape[i], hi.shape[i])
        assert_same_matrices(n, ad @ a)
        assert_same_matrices(ad, a.conjugate().transpose())

    for i in range(3):
        print("i=", i)
        a = bdestroy(hi, i)
        ad = bcreate(hi, i)
        n = bnumber(hi, i)
        for j in range(3, 5):
            print("j=", i)
            a = bdestroy(hi, j)
            ad = bcreate(hi, j)
            n = bnumber(hi, j)

        assert_same_matrices(n, ad @ a)
        assert_same_matrices(ad, a.conjugate().transpose())


def test_mul_matmul():
    hi = nk.hilbert.Spin(s=1 / 2, N=2)
    sx0_hat = nk.operator.LocalOperator(hi, sx, [0])
    sy1_hat = nk.operator.LocalOperator(hi, sy, [1])

    sx0sy1_hat = sx0_hat @ sy1_hat
    assert_same_matrices(sx0sy1_hat.to_dense(), sx0_hat.to_dense() @ sy1_hat.to_dense())
    sx0sy1_hat = sx0_hat * sy1_hat
    assert_same_matrices(sx0sy1_hat.to_dense(), sx0_hat.to_dense() @ sy1_hat.to_dense())

    op = nk.operator.LocalOperator(hi, sx, [0])
    with raises(ValueError):
        op @= nk.operator.LocalOperator(hi, sy, [1])

    op = nk.operator.LocalOperator(hi, sx, [0], dtype=complex)
    op @= nk.operator.LocalOperator(hi, sy, [1])
    assert_same_matrices(op.to_dense(), sx0sy1_hat.to_dense())

    op = nk.operator.LocalOperator(hi, sx, [0], dtype=complex)
    op *= nk.operator.LocalOperator(hi, sy, [1])
    assert_same_matrices(op.to_dense(), sx0sy1_hat.to_dense())

    assert_same_matrices((2.0 * sx0sy1_hat).to_dense(), 2.0 * sx0sy1_hat.to_dense())
    assert_same_matrices((sx0sy1_hat * 2.0).to_dense(), 2.0 * sx0sy1_hat.to_dense())

    op *= 2.0
    assert_same_matrices(op.to_dense(), 2.0 * sx0sy1_hat.to_dense())

    with pytest.raises(TypeError):
        sx0_hat @ 2.0
    with pytest.raises(TypeError):
        op = nk.operator.LocalOperator(hi, sx, [0])
        op @= 2.0


def test_complicated_mul():
    # If this test fails probably we are tripping the reordering
    L = 5  # 10
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

    ha = nk.operator.Ising(hi, graph=g, h=0.4)

    assert_same_matrices(ha.to_dense(), ha.to_local_operator().to_dense())
    assert_same_matrices(ha.to_dense() @ ha.to_dense(), (ha @ ha).to_dense())


def test_truediv():
    hi = nk.hilbert.Spin(s=1 / 2, N=2)

    sx0_hat = nk.operator.LocalOperator(hi, sx, [0])
    sy1_hat = nk.operator.LocalOperator(hi, sy, [1])
    sx0sy1_hat = sx0_hat @ sy1_hat

    assert_same_matrices((sx0sy1_hat / 2.0).to_dense(), sx0sy1_hat.to_dense() / 2.0)
    assert_same_matrices((sx0sy1_hat / 2.0).to_dense(), 0.5 * sx0sy1_hat.to_dense())

    assert_same_matrices((sx0sy1_hat / 2).to_dense(), sx0sy1_hat.to_dense() / 2)
    assert_same_matrices((sx0sy1_hat / 2).to_dense(), 0.5 * sx0sy1_hat.to_dense())

    with pytest.raises(TypeError):
        sx0_hat / sy1_hat
    with pytest.raises(TypeError):
        1.0 / sx0_hat

    sx0sy1 = sx0sy1_hat.to_dense()
    sx0sy1_hat /= 3.0
    assert_same_matrices(sx0sy1_hat.to_dense(), sx0sy1 / 3.0)


@pytest.mark.parametrize(
    "op",
    [pytest.param(op, id=name) for name, op in herm_operators.items()],
)
def test_copy(op):
    op_copy = op.copy()
    assert op_copy is not op
    for o1, o2 in zip(op._operators, op_copy._operators):
        # assert o1 is not o2
        assert_same_matrices(o1, o2)
    assert_same_matrices(op, op_copy)


def test_type_promotion():
    hi = nk.hilbert.Qubit(1)
    real_op = nk.operator.spin.sigmax(hi, 0, dtype=float)
    complex_mat = nk.operator.spin.sigmay(hi, 0, dtype=complex).to_dense()
    promoted_op = real_op + nk.operator.LocalOperator(hi, complex_mat, acting_on=[0])
    assert promoted_op.dtype == np.complex128

    op = nk.operator.spin.sigmax(hi, 0, dtype=np.float32)
    op2 = 2 * op
    assert op2.dtype == np.float32
    op2 = 2.0 * op
    assert op2.dtype == np.float32
    op2 = 2.0j * op
    assert op2.dtype == np.complex64

    op2 = 2 - op
    assert op2.dtype == np.float32
    op2 = 2.0 - op
    assert op2.dtype == np.float32
    op2 = 2.0j - op
    assert op2.dtype == np.complex64

    op = nk.operator.spin.sigmay(hi, 0, dtype=np.complex64)
    op2 = 2 * op
    assert op2.dtype == np.complex64
    op2 = 2.0 * op
    assert op2.dtype == np.complex64


def test_empty_after_sum():
    a = nk.operator.spin.sigmaz(nk.hilbert.Spin(0.5), 0)
    zero_op = a - a
    np.testing.assert_allclose(zero_op.to_dense(), 0.0)

    a = nk.operator.spin.sigmay(nk.hilbert.Spin(0.5), 0)
    zero_op = a - a
    np.testing.assert_allclose(zero_op.to_dense(), 0.0)


@pytest.mark.parametrize(
    "op",
    [pytest.param(op, id=name) for name, op in herm_operators.items()],
)
def test_is_hermitian(op):
    assert op.is_hermitian

    op2 = 1j * op
    assert not op2.is_hermitian


@pytest.mark.parametrize(
    "ops",
    [pytest.param(op, id=name) for name, op in generic_operators.items()],
)
def test_is_hermitian_generic_op(ops):
    op, oph = ops

    assert not op.is_hermitian
    assert not oph.is_hermitian


@pytest.mark.parametrize(
    "jax",
    [pytest.param(op) for op in [True, False]],
)
@common.skipif_sharding
def test_qutip_conversion(jax):
    # skip test if qutip not installed
    pytest.importorskip("qutip")

    hi = nk.hilbert.Spin(s=1 / 2, N=2)
    op = nk.operator.spin.sigmax(hi, 0)
    if jax:
        op = op.to_jax_operator()

    q_obj = op.to_qobj()

    assert q_obj.type == "oper"
    assert len(q_obj.dims) == 2
    assert q_obj.dims[0] == list(op.hilbert.shape)
    assert q_obj.dims[1] == list(op.hilbert.shape)

    assert q_obj.shape == (op.hilbert.n_states, op.hilbert.n_states)
    np.testing.assert_allclose(q_obj.data.to_array(), op.to_dense())


def test_notsharing():
    # This test will fail if operators alias some underlying arrays upon copy().
    hi = nk.hilbert.Spin(0.5, 2)
    a = nk.operator.spin.sigmax(hi, 0) * nk.operator.spin.sigmax(hi, 1, dtype=complex)
    b = nk.operator.spin.sigmay(hi, 0) * nk.operator.spin.sigmaz(hi, 1)
    delta = b - a

    a_orig = a.to_dense()
    a_copy = a.copy()
    a_copy += delta

    np.testing.assert_allclose(a_orig, a.to_dense())
    np.testing.assert_allclose(a_copy.to_dense(), b.to_dense())


def test_correct_minus():
    # at some point during the rewrite this got broken
    hi = nk.hilbert.Fock(3)
    n = nk.operator.boson.number(hi, 0)
    nd = n.to_dense()
    I = np.eye(4)

    op = n @ (n - 1)
    op2 = (n - 1) @ n
    opd = nd @ (nd - I)

    # they commute
    assert_same_matrices(op, op2)
    # they commute
    assert_same_matrices(op, opd)


def test_operator():
    # check that heterogeneous hilbert spaces are ordered correctly #1106
    n_max = 5
    hi = nk.hilbert.Fock(n_max, N=1) * nk.hilbert.Qubit()
    a = nk.operator.boson.destroy(hi, 0)
    sp = nk.operator.spin.sigmap(hi, 1)
    op1 = a * sp
    op2 = sp * a

    assert_same_matrices(op1, op2)


def test_error_if_wrong_shape():
    # Issue #1157
    # https://github.com/netket/netket/issues/1157
    hi = nk.hilbert.Fock(5, N=3)
    mat = np.random.rand(3, 3)
    with pytest.raises(ValueError):
        nk.operator.LocalOperator(hi, mat, [0, 1])


def test_inhomogeneous_hilb_issue_1192():
    # Issue #1192
    # https://github.com/netket/netket/issues/1192
    hi = nk.hilbert.Fock(n_max=3) * nk.hilbert.Spin(1 / 2) * nk.hilbert.Fock(n_max=2)
    c0 = bcreate(hi, 0)
    d2 = bdestroy(hi, 2)

    assert_same_matrices(c0 @ d2, c0.to_dense() @ d2.to_dense())


def test_add_transpose():
    hi = nk.hilbert.Fock(n_max=3)
    c0 = bcreate(hi, 0)
    assert_same_matrices(c0 + c0.H, c0.to_dense() + c0.H.to_dense())


def test_identity():
    hi = nk.hilbert.Fock(n_max=3)
    I = nk.operator.LocalOperator(hi, constant=1)

    assert_same_matrices(I, np.eye(hi.n_states))

    X = bcreate(hi, 0)
    assert_same_matrices(I @ X, X)


@common.skipif_sharding
def test_not_recompiling():
    hi = nk.hilbert.Fock(n_max=3) * nk.hilbert.Spin(1 / 2) * nk.hilbert.Fock(n_max=2)
    op = bcreate(hi, 0) * bdestroy(hi, 2)

    assert not op._initialized
    op.get_conn_padded(hi.numbers_to_states(1))
    assert op._initialized


def test_duplicate_sites():
    hi = nk.hilbert.Spin(s=1 / 2, N=2)
    mat = np.random.rand(4, 4)
    # The operator at index 0 acts on duplicated sites [0, 0]
    with raises(ValueError):
        nk.operator.LocalOperator(hi, mat, [0, 0])


def test_numpy_matrix():
    # np.matrix dont respect the API of ndarray. They
    # must be specially handled
    hi = nk.hilbert.Spin(0.5, 1)
    mat = np.matrix([[1, 0], [0, 1]])
    op = nk.operator.LocalOperator(hi, mat, 0)
    assert_same_matrices(mat, op)


def test_mixed_sparse_dense_terms():
    # bug #1596
    # tests for https://github.com/netket/netket/issues/1596

    hi = nk.hilbert.Spin(0.5, 3)
    op = nk.operator.spin.sigmax(hi, 0)
    mat = op.operators[0].todense()

    op2 = nk.operator.LocalOperator(hi, mat, 1)
    op3 = op - op2

    res = op3 @ op3
    res2 = op3.to_pauli_strings() @ op3.to_pauli_strings()
    assert_same_matrices(res, res2)


def test_pauli_strings_conversion():
    hi = nk.hilbert.Spin(1 / 2, N=5)

    op_dict = {
        "X": lambda idx: nk.operator.spin.sigmax(hi, idx, dtype=complex),
        "Y": lambda idx: nk.operator.spin.sigmay(hi, idx, dtype=complex),
        "Z": lambda idx: nk.operator.spin.sigmaz(hi, idx, dtype=complex),
    }

    def _convert(operators, weights, constant):
        ps = nk.operator.PauliStrings(
            hi, operators + ["I" * hi.size], weights + [constant], dtype=complex
        )
        lo = nk.operator.LocalOperator(hi, dtype=complex)
        for op, w in zip(operators, weights):
            _lo = None
            for i, gate in enumerate(op):
                if gate == "I":
                    continue
                gate_lo = op_dict[gate](i)
                if _lo is None:
                    _lo = gate_lo
                else:
                    _lo = _lo @ gate_lo
            if _lo is None:  # identity
                lo += w
            else:
                lo += w * _lo
        lo += constant
        ps_conv = lo.to_pauli_strings()
        return ps, lo, ps_conv

    operators = ["ZIIII", "IIIYI", "IIXII", "IIIII"]
    weights = [1.0, 1j, 3, 1]

    ps_true, lo, ps_conv = _convert(operators, weights, 9.0)
    np.testing.assert_allclose(ps_true.to_dense(), ps_conv.to_dense())
    np.testing.assert_allclose(lo.to_dense(), ps_conv.to_dense())

    operators = [
        "IZZII",
        "IZYII",
        "IIIIX",
        "IIZZY",
        "IIZXI",
        "IIZZI",
        "IXYXY",
    ]
    weights = [1.0, 1j, 3, 1, 7, 9.5, 6.6]

    ps_true, lo, ps_conv = _convert(operators, weights, 1.1)
    np.testing.assert_allclose(ps_true.to_dense(), ps_conv.to_dense())
    np.testing.assert_allclose(lo.to_dense(), ps_conv.to_dense())


def test_pauli_strings_conversion_no_warn():
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=ComplexWarning)
        nk.operator.spin.sigmax(nk.hilbert.Spin(0.5, 3), 0).to_pauli_strings()

    with pytest.raises(
        TypeError, match=r".* hilbert spaces with local dimension != 2.*"
    ):
        nk.operator.spin.sigmax(nk.hilbert.Spin(1.0, 3), 0).to_pauli_strings()
