import numpy as np
import pytest

import netket as nk
from netket.experimental.operator import ParticleNumberConservingFermioperator2nd


operator_classes = pytest.mark.parametrize(
    "operator_cls", [nk.operator.LocalOperatorNumba, nk.operator.LocalOperatorJax]
)


@operator_classes
def test_adjoint_matmul_unconstrained_is_squared(operator_cls):
    hi = nk.hilbert.Fock(n_max=2, N=2)
    a = nk.operator.boson.destroy(hi, 0, cls=operator_cls)

    squared = a.H @ a
    assert isinstance(squared, nk.operator.Squared)
    np.testing.assert_allclose(
        squared.to_dense(), nk.operator.boson.number(hi, 0).to_dense()
    )


@operator_classes
def test_adjoint_matmul_constrained_is_product(operator_cls):
    # a does not preserve the particle number, so a.H @ a must not be Squared(a),
    # whose |a_loc|² estimator reads the wavefunction outside of the constraint.
    hi = nk.hilbert.Fock(n_max=2, N=2, n_particles=1)
    a = nk.operator.boson.destroy(hi, 0, cls=operator_cls)

    product = a.H @ a
    assert not isinstance(product, nk.operator.Squared)
    np.testing.assert_allclose(
        product.to_dense(), nk.operator.boson.number(hi, 0).to_dense()
    )


def test_adjoint_matmul_constrained_without_adjoint():
    # The adjoint of this operator cannot be constructed explicitly, so on a
    # constrained space A.H @ A raises instead of silently returning Squared(A).
    hi = nk.hilbert.SpinOrbitalFermions(3, n_fermions=1)
    c = nk.operator.fermion.create(hi, 0) @ nk.operator.fermion.destroy(hi, 1)
    pnc = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(c)
    assert not pnc.is_hermitian

    with pytest.raises(NotImplementedError, match="Squared"):
        pnc.H @ pnc


def test_explicit_squared_constrained():
    # An explicit Squared(A) is kept on constrained spaces, and is exact if A
    # preserves the constraint, even when A†A cannot be formed explicitly.
    hi = nk.hilbert.SpinOrbitalFermions(3, n_fermions=1)
    c = nk.operator.fermion.create(hi, 0) @ nk.operator.fermion.destroy(hi, 1)
    pnc = ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd(c)

    squared = nk.operator.Squared(pnc)
    expected = (c.H.collect() @ c).to_dense()
    np.testing.assert_allclose(squared.to_dense(), expected)
    np.testing.assert_allclose(squared.to_sparse().toarray(), expected)
