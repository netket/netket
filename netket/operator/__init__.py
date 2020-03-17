from __future__ import absolute_import
from .._C_netket.operator import *

from . import spin, boson

from .local_values import (
    local_values,
    der_local_values,
)

from .hamiltonian import (
    Ising,
    Heisenberg,
)

from .._C_netket.operator import _rotated_grad_kernel
