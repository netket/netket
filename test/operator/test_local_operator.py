from netket.operator.boson import (
    create as bcreate,
    destroy as bdestroy,
    number as bnumber,
)
from netket.operator.spin import sigmax, sigmay, sigmaz, sigmam, sigmap
from netket.operator import LocalOperator
import netket as nk
import numpy as np
import pytest
from pytest import approx, raises

herm_operators = {}
generic_operators = {}

# Custom Hamiltonian
sx = [[0, 1], [1, 0]]
sy = [[0, -1.0j], [1.0j, 0]]
sz = [[1, 0], [0, -1]]
sm = [[0, 0], [1, 0]]
sp = [[0, 1], [0, 0]]
g = nk.graph.Graph(edges=[[i, i + 1] for i in range(8)])
hi = nk.hilbert.CustomHilbert(local_states=[-1, 1], N=g.n_nodes)


def _loc(*args):
    return nk.operator.LocalOperator(hi, *args)


sx_hat = _loc([sx] * 3, [[0], [1], [5]])
sy_hat = _loc([sy] * 4, [[2], [3], [4], [7]])
szsz_hat = _loc(sz, [0]) @ _loc(sz, [1])
szsz_hat += _loc(sz, [4]) @ _loc(sz, [5])
szsz_hat += _loc(sz, [6]) @ _loc(sz, [8])
szsz_hat += _loc(sz, [7]) @ _loc(sz, [0])

herm_operators["sx (real op)"] = sx_hat
herm_operators["sy"] = sy_hat

herm_operators["Custom Hamiltonian"] = sx_hat + sy_hat + szsz_hat
herm_operators["Custom Hamiltonian Prod"] = sx_hat * 1.5 + (2.0 * sy_hat)

sm_hat = nk.operator.LocalOperator(hi, [sm] * 3, [[0], [1], [4]])
sp_hat = nk.operator.LocalOperator(hi, [sp] * 3, [[0], [1], [4]])


generic_operators["sigma +/-"] = (sm_hat, sp_hat)


def same_matrices(matl, matr, eps=1.0e-6):
    if isinstance(matl, LocalOperator):
        matl = matl.to_dense()

    if isinstance(matr, LocalOperator):
        matr = matr.to_dense()

    assert np.max(np.abs(matl - matr)) == approx(0.0, rel=eps, abs=eps)


def test_hermitian_local_operator_transpose_conjugation():
    for name, op in herm_operators.items():
        orig_op = op.copy()

        op_t = op.transpose()
        op_c = op.conjugate()
        op_h = op.transpose().conjugate()

        assert [
            same_matrices(m1, m2) for (m1, m2) in zip(op._operators, orig_op._operators)
        ]

        mat = op.to_dense()
        mat_t = op_t.to_dense()
        mat_c = op_c.to_dense()
        mat_h = op_h.to_dense()

        assert [
            same_matrices(m1, m2) for (m1, m2) in zip(op._operators, orig_op._operators)
        ]

        same_matrices(mat, mat_h)
        same_matrices(mat_t, mat_c)

        mat_t_t = op.transpose().transpose().to_dense()
        mat_c_c = op.conjugate().conjugate().to_dense()

        same_matrices(mat, mat_t_t)
        same_matrices(mat, mat_c_c)

        assert [
            same_matrices(m1, m2) for (m1, m2) in zip(op._operators, orig_op._operators)
        ]


def test_local_operator_transpose_conjugation():
    for name, (op, oph) in generic_operators.items():

        mat = op.to_dense()
        math = oph.to_dense()

        mat_h = op.transpose().conjugate().to_dense()
        same_matrices(mat_h, math)

        math_h = oph.transpose().conjugate().to_dense()
        same_matrices(math_h, mat)


def test_local_operator_add():
    sz0 = nk.operator.spin.sigmaz(hi, 0)

    ham = 0.5 * sz0.to_dense()
    ha = 0.5 * sz0
    ha2 = nk.operator.spin.sigmaz(hi, 0)
    ha2 *= 0.5
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    ha = ha * 1j
    with raises(ValueError):
        ha2 *= 1j

    ha2 = ha2 * 1j
    ham = ham * 1j
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    for i in range(1, 3):
        ha = ha + 0.2 * nk.operator.spin.sigmaz(hi, i)
        ha2 += 0.2 * nk.operator.spin.sigmaz(hi, i)
        ham += 0.2 * nk.operator.spin.sigmaz(hi, i).to_dense()
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    for i in range(3, 5):
        ha = ha + 0.2 * nk.operator.spin.sigmax(hi, i)
        ha2 += 0.2 * nk.operator.spin.sigmax(hi, i)
        ham += 0.2 * nk.operator.spin.sigmax(hi, i).to_dense()
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    for i in range(5, 7):
        ha = ha - 0.3 * nk.operator.spin.sigmam(hi, i)
        ha2 -= 0.3 * nk.operator.spin.sigmam(hi, i)
        ham -= 0.3 * nk.operator.spin.sigmam(hi, i).to_dense()
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    ha = ha - 0.3j * nk.operator.spin.sigmam(hi, 7)
    ha2 -= 0.3j * nk.operator.spin.sigmam(hi, 7)
    ham -= 0.3j * nk.operator.spin.sigmam(hi, 7).to_dense()
    same_matrices(ha, ha2)
    same_matrices(ha, ham)

    # test commutativity
    ha = LocalOperator(hi)
    ha2 = LocalOperator(hi, dtype=complex)
    for i in range(0, 3):
        ha += 0.3 * nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmax(hi, i + 1)
        ha += 0.4 * nk.operator.spin.sigmaz(hi, i)
        ha2 += 0.5 * nk.operator.spin.sigmay(hi, i)

    ha_ha2 = ha + ha2
    ha2_ha = ha2 + ha
    same_matrices(ha_ha2, ha2_ha)


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
        assert (sigmax(hi, i).to_dense() == sx_hat.to_dense()).all()
        assert (sigmay(hi, i).to_dense() == sy_hat.to_dense()).all()
        assert (sigmaz(hi, i).to_dense() == sz_hat.to_dense()).all()

    print("Testing Sigma_+/-...")
    for i in range(L):
        sm_hat = nk.operator.LocalOperator(hi, sm, [i])
        sp_hat = nk.operator.LocalOperator(hi, sp, [i])
        assert (sigmam(hi, i).to_dense() == sm_hat.to_dense()).all()
        assert (sigmap(hi, i).to_dense() == sp_hat.to_dense()).all()

    print("Testing Sigma_+/- composition...")

    hi = nk.hilbert.Spin(0.5, N=L)
    for i in range(L):
        sx = sigmax(hi, i)
        sy = sigmay(hi, i)
        sigmam_hat = 0.5 * (sx + (-1j) * sy)
        sigmap_hat = 0.5 * (sx + (1j) * sy)
        assert (sigmam(hi, i).to_dense() == sigmam_hat.to_dense()).all()
        assert (sigmap(hi, i).to_dense() == sigmap_hat.to_dense()).all()

    print("Testing create/destroy composition...")
    hi = nk.hilbert.Fock(3, N=L)
    for i in range(L):
        print("i=", i)
        a = bdestroy(hi, i)
        ad = bcreate(hi, i)
        n = bnumber(hi, i)

        assert np.allclose(n.to_dense(), (ad @ a).to_dense())
        assert (ad.to_dense() == a.conjugate().transpose().to_dense()).all()

    print("Testing mixed spaces...")
    L = 3
    his = nk.hilbert.Spin(0.5, N=L)
    hib = nk.hilbert.Fock(3, N=L - 1)
    hi = his * hib
    for i in range(hi.size):
        print("i=", i)
        sx = sigmax(hi, i)

        assert sx.operators[0].shape == (hi.shape[i], hi.shape[i])
        assert np.allclose(n.to_dense(), (ad @ a).to_dense())
        assert (ad.to_dense() == a.conjugate().transpose().to_dense()).all()

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

        assert np.allclose(n.to_dense(), (ad @ a).to_dense())
        assert (ad.to_dense() == a.conjugate().transpose().to_dense()).all()


def test_mul_matmul():
    hi = nk.hilbert.Spin(s=1 / 2, N=2)
    sx0_hat = nk.operator.LocalOperator(hi, sx, [0])
    sy1_hat = nk.operator.LocalOperator(hi, sy, [1])

    sx0sy1_hat = sx0_hat @ sy1_hat
    assert np.allclose(sx0sy1_hat.to_dense(), sx0_hat.to_dense() @ sy1_hat.to_dense())
    sx0sy1_hat = sx0_hat * sy1_hat
    assert np.allclose(sx0sy1_hat.to_dense(), sx0_hat.to_dense() @ sy1_hat.to_dense())

    op = nk.operator.LocalOperator(hi, sx, [0])
    with raises(ValueError):
        op @= nk.operator.LocalOperator(hi, sy, [1])

    op = nk.operator.LocalOperator(hi, sx, [0], dtype=complex)
    op @= nk.operator.LocalOperator(hi, sy, [1])
    assert np.allclose(op.to_dense(), sx0sy1_hat.to_dense())

    op = nk.operator.LocalOperator(hi, sx, [0], dtype=complex)
    op *= nk.operator.LocalOperator(hi, sy, [1])
    assert np.allclose(op.to_dense(), sx0sy1_hat.to_dense())

    assert np.allclose((2.0 * sx0sy1_hat).to_dense(), 2.0 * sx0sy1_hat.to_dense())
    assert np.allclose((sx0sy1_hat * 2.0).to_dense(), 2.0 * sx0sy1_hat.to_dense())

    op *= 2.0
    assert np.allclose(op.to_dense(), 2.0 * sx0sy1_hat.to_dense())

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

    assert np.allclose(ha.to_dense(), ha.to_local_operator().to_dense())
    assert np.allclose(ha.to_dense() @ ha.to_dense(), (ha @ ha).to_dense())


def test_truediv():
    hi = nk.hilbert.Spin(s=1 / 2, N=2)

    sx0_hat = nk.operator.LocalOperator(hi, sx, [0])
    sy1_hat = nk.operator.LocalOperator(hi, sy, [1])
    sx0sy1_hat = sx0_hat @ sy1_hat

    assert np.allclose((sx0sy1_hat / 2.0).to_dense(), sx0sy1_hat.to_dense() / 2.0)
    assert np.allclose((sx0sy1_hat / 2.0).to_dense(), 0.5 * sx0sy1_hat.to_dense())

    assert np.allclose((sx0sy1_hat / 2).to_dense(), sx0sy1_hat.to_dense() / 2)
    assert np.allclose((sx0sy1_hat / 2).to_dense(), 0.5 * sx0sy1_hat.to_dense())

    with pytest.raises(TypeError):
        sx0_hat / sy1_hat
    with pytest.raises(TypeError):
        1.0 / sx0_hat

    sx0sy1 = sx0sy1_hat.to_dense()
    sx0sy1_hat /= 3.0
    assert np.allclose(sx0sy1_hat.to_dense(), sx0sy1 / 3.0)


def test_copy():
    for name, op in herm_operators.items():
        print(name)
        print(op)
        op_copy = op.copy()
        assert op_copy is not op
        for o1, o2 in zip(op._operators, op_copy._operators):
            assert o1 is not o2
            assert np.all(o1 == o2)
        same_matrices(op, op_copy)


def test_raises_unsorted_hilbert():
    hi = nk.hilbert.CustomHilbert([-1, 1, 0], N=3)
    with pytest.raises(ValueError):
        nk.operator.LocalOperator(hi)
