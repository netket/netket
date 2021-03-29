from .abstract_optimizer import AbstractOptimizer

from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .ada_max import AdaMax
from .ams_grad import AmsGrad
from .momentum import Momentum
from .rms_prop import RmsProp
from .sgd import Sgd
from .stochastic_reconfiguration import SR

from netket.utils import (
    jax_available as _jax_available,
    torch_available as _torch_available,
)

if _jax_available:
    from . import jax

if _torch_available:
    from .torch import Torch
