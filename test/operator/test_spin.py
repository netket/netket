import pytest

from netket.operator import spin
import netket as nk
import numpy as np
from numpy.testing import assert_almost_equal

herm_operators = {}
generic_operators = {}

N = 4


@pytest.mark.parametrize("S", [1 / 2, 1, 3 / 2])
def test_pauli_algebra(S):
    hi = nk.hilbert.Spin(S) ** N

    for i in range(N):
        sx = spin.sigmax(hi, i)
        sy = spin.sigmay(hi, i)
        sz = spin.sigmaz(hi, i)

        sm = spin.sigmam(hi, i)
        sp = spin.sigmap(hi, i)

        assert_almost_equal(0.5 * (sx - 1j * sy).to_dense(), sm.to_dense())
        assert_almost_equal(0.5 * (sx + 1j * sy).to_dense(), sp.to_dense())
        assert_almost_equal(0.5 * (sx.to_dense() - 1j * sy.to_dense()), sm.to_dense())
        assert_almost_equal(0.5 * (sx.to_dense() + 1j * sy.to_dense()), sp.to_dense())

        if S == 1 / 2:
            Imat = np.eye(hi.n_states)

            # check that -i sx sy sz = I
            assert_almost_equal((-1j * sx @ sy @ sz).to_dense(), Imat)
            assert_almost_equal(
                (-1j * sx.to_dense() @ sy.to_dense() @ sz.to_dense()), Imat
            )
