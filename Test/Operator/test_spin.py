from netket.operator.boson import (
    create as bcreate,
    destroy as bdestroy,
    number as bnumber,
)
from netket.operator import spin, boson
from netket.operator import LocalOperator
import netket.legacy as nk
import networkx as nx
import numpy as np
import pytest
from pytest import approx, raises
import os

from numpy import testing
from numpy.testing import assert_almost_equal

herm_operators = {}
generic_operators = {}

N = 4


def test_pauli_algebra():
    hi = nk.hilbert.Spin(1 / 2) ** N

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

        Imat = np.eye(hi.n_states)

        # check that -i sx sy sz = I
        assert_almost_equal((-1j * sx @ sy @ sz).to_dense(), Imat)
        assert_almost_equal((-1j * sx.to_dense() @ sy.to_dense() @ sz.to_dense()), Imat)
