import pytest
from pytest import approx
import netket as nk
import numpy as np

from .. import common

pytestmark = common.skipif_distributed


operators = {}

g = nk.graph.Chain(8)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
operators["Ising 1D"] = nk.operator.Ising(graph=g, h=1.0, hilbert=hi)
operators["Ising 1D Jax"] = nk.operator.IsingJax(graph=g, h=1.0, hilbert=hi)


@pytest.mark.parametrize(
    "matrix_free", [pytest.param(x, id=f"matrix_free={x}") for x in [False, True]]
)
@pytest.mark.parametrize(
    "ha", [pytest.param(op, id=name) for name, op in operators.items()]
)
def test_ed(ha, matrix_free):
    first_n = 3

    def expval(op, v):
        return np.vdot(v, op(v))

    # Test Lanczos ED with eigenvectors
    w, v = nk.exact.lanczos_ed(
        ha, k=first_n, compute_eigenvectors=True, matrix_free=matrix_free
    )
    assert w.shape == (first_n,)
    assert v.shape == (hi.n_states, first_n)
    gse = expval(ha, v[:, 0])
    fse = expval(ha, v[:, 1])
    assert gse == approx(w[0], rel=1e-14, abs=1e-14)
    assert fse == approx(w[1], rel=1e-14, abs=1e-14)

    # Test Lanczos ED without eigenvectors
    w = nk.exact.lanczos_ed(
        ha, k=first_n, compute_eigenvectors=False, matrix_free=matrix_free
    )
    assert w.shape == (first_n,)

    # Test Lanczos ED with custom options
    w_tol = nk.exact.lanczos_ed(
        ha,
        k=first_n,
        scipy_args={"tol": 1e-9, "maxiter": 1000},
        matrix_free=matrix_free,
    )
    assert w_tol.shape == (first_n,)
    assert w_tol == approx(w)

    # Test Full ED with eigenvectors
    w_full, v_full = nk.exact.full_ed(ha, compute_eigenvectors=True)
    assert w_full.shape == (hi.n_states,)
    assert v_full.shape == (hi.n_states, hi.n_states)
    gse = expval(ha, v_full[:, 0])
    fse = expval(ha, v_full[:, 1])
    assert gse == approx(w_full[0], rel=1e-14, abs=1e-14)
    assert fse == approx(w_full[1], rel=1e-14, abs=1e-14)
    assert w == approx(w_full[:3], rel=1e-14, abs=1e-14)

    # Test Full ED without eigenvectors
    w_full = nk.exact.full_ed(ha, compute_eigenvectors=False)
    assert w_full.shape == (hi.n_states,)
    assert w == approx(w_full[:3], rel=1e-14, abs=1e-14)


def test_ed_restricted():
    g = nk.graph.Hypercube(length=8, n_dim=1, pbc=True)
    hi1 = nk.hilbert.Spin(s=0.5, N=g.n_nodes, total_sz=0)
    hi2 = nk.hilbert.Spin(s=0.5, N=g.n_nodes)

    ham1 = nk.operator.Heisenberg(hi1, graph=g)
    ham2 = nk.operator.Heisenberg(hi2, graph=g)

    assert ham1.to_linear_operator().shape == (70, 70)
    assert ham2.to_linear_operator().shape == (256, 256)

    w1, v1 = nk.exact.lanczos_ed(ham1, compute_eigenvectors=True)
    w2, v2 = nk.exact.lanczos_ed(ham2, compute_eigenvectors=True)

    assert w1[0] == approx(w2[0])

    def overlap(phi, psi):
        bare_overlap = np.abs(np.vdot(phi, psi)) ** 2
        return bare_overlap / (np.vdot(phi, phi) * np.vdot(psi, psi)).real

    # Non-zero elements of ground state in full Hilbert space should equal the ground
    # state in the constrained Hilbert space
    idx_nonzero = np.abs(v2[:, 0]) > 1e-4
    assert overlap(v1[:, 0], v2[:, 0][idx_nonzero]) == approx(1.0)
