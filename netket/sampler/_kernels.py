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


# Register Jax kernels here
from ..utils import jax_available

if jax_available:
    from . import jax
    from ..machine import Jax as JaxMachine

    @_LocalKernel.register(JaxMachine)
    def _Jax_LocalKernel(machine):
        return jax._LocalKernel(machine.hilbert.local_states, machine.input_size)

    @_ExchangeKernel.register(JaxMachine)
    def _Jax_ExchangeKernel(machine, d_max):
        return jax._ExchangeKernel(machine.hilbert, d_max)

    @_HamiltonianKernel.register(JaxMachine)
    def _Jax_HamiltonianKernel(machine, hamiltonian):
        raise NotImplementedError

    @_CustomKernel.register(JaxMachine)
    def _Jax_CustomKernel(machine, move_operators, move_weights=None):
        raise NotImplementedError


# Register PyTorch kernels here
