from __future__ import absolute_import
from .._C_netket.operator import *

from . import spin, boson

from .local_values import (
    local_values,
    der_local_values,
)


from .local_operator import PyLocalOperator
from .graph_operator import PyGraphOperator

from .hamiltonian import (
    Ising,
    Heisenberg,
    PyIsing,
    PyHeisenberg
)

from .._C_netket.operator import _rotated_grad_kernel
