from functools import singledispatch
import numpy as _np
from . import numpy

# Register numpy kernels here
@singledispatch
def _LocalKernel(machine):
    return numpy._LocalKernel(
        _np.asarray(machine.hilbert.local_states), machine.input_size
    )


@singledispatch
def _ExchangeKernel(machine, d_max):
    return numpy._ExchangeKernel(machine.hilbert, d_max)


@singledispatch
def _CustomKernel(machine, move_operators, move_weights=None):
    return numpy._CustomKernel(move_operators, move_weights)


@singledispatch
def _HamiltonianKernel(machine, hamiltonian):
    return numpy._HamiltonianKernel(hamiltonian)
