from .abstract_machine import AbstractMachine

from .rbm import RbmSpin, RbmSpinReal, RbmSpinSymm, RbmMultiVal, RbmSpinPhase
from .jastrow import Jastrow, JastrowSymm
from ..utils import jax_available, torch_available


if jax_available:
    from .jax import Jax, JaxRbm, JaxMpsPeriodic


if torch_available:
    from .torch import Torch, TorchLogCosh, TorchView


# def MPSPeriodicDiagonal(hilbert, bond_dim, symperiod=-1):
#     return MPSPeriodic(hilbert, bond_dim, diag=True, symperiod=symperiod)


from . import density_matrix
