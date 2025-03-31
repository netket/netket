import pytest
import numpy as np
from netket.operator import to_quspin_format, target_symmetry_subsector, Ising
from netket.hilbert import Spin

def test_to_quspin_format():
    hilbert = Spin(s=0.5, N=10)
    operator = Ising(hilbert, h=1.0)
    quspin_operator = to_quspin_format(operator)
    assert quspin_operator is not None

def test_target_symmetry_subsector():
    hilbert = Spin(s=0.5, N=10)
    operator = Ising(hilbert, h=1.0)
    with pytest.raises(NotImplementedError):
        target_symmetry_subsector(operator, subsector="translation")
