from .abstract_machine import AbstractMachine

from .rbm import RbmSpin, RbmSpinReal, RbmSpinSymm, RbmMultiVal, RbmSpinPhase
from .jastrow import Jastrow, JastrowSymm
from ..utils import jax_available, torch_available, flax_available


if jax_available:
    from .jax import Jax, JaxRbm, MPSPeriodic, JaxRbmSpinPhase
    from .jax import DenseReal, SumLayer, LogCoshLayer

    if flax_available:
        from .flax import Flax

if torch_available:
    from .torch import Torch, TorchLogCosh, TorchView


from . import density_matrix
