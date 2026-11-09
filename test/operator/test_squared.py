import numpy as np
import pytest

import netket as nk


@pytest.mark.parametrize(
    "operator_cls", [nk.operator.LocalOperatorNumba, nk.operator.LocalOperatorJax]
)
@pytest.mark.parametrize("n_particles", [None, 1])
def test_squared_matrix(operator_cls, n_particles):
    hi = nk.hilbert.Fock(n_max=2, N=1, n_particles=n_particles)
    a = nk.operator.boson.destroy(hi, 0, dtype=complex, cls=operator_cls)
    squared = a.H @ a
    expected = np.diag([0.0, 1.0, 2.0]) if n_particles is None else np.ones((1, 1))

    np.testing.assert_allclose(squared.to_dense(), expected)
    np.testing.assert_allclose(squared.to_sparse().toarray(), expected)
