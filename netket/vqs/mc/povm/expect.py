from netket.utils.dispatch import dispatch
from netket.vqs.mc import (
    kernels,
    check_hilbert,
    get_local_kernel_arguments,
    get_local_kernel,
)
from netket.operator import Squared

from .state import MCPOVMState


@dispatch
def get_local_kernel_arguments(vstate: MCPOVMState, Ô: Squared):  # noqa: F811
    check_hilbert(vstate.hilbert, Ô.hilbert)

    σ = vstate.samples
    σp, mels = Ô.parent.get_conn_padded(σ)
    return σ, (σp, mels)


@dispatch
def get_local_kernel(vstate: MCPOVMState, Ô: Squared):  # noqa: F811
    return kernels.local_value_squared_kernel
