from .abstract_optimizer import AbstractOptimizer

from .ada_delta import AdaDelta
from .ada_grad import AdaGrad
from .ada_max import AdaMax
from .ams_grad import AmsGrad
from .momentum import Momentum
from .rms_prop import RmsProp
from .sgd import Sgd

from .stochastic_reconfiguration import SR

from ..utils import jax_available, torch_available

if jax_available:
    from .jax import Jax
    from .jax_stochastic_reconfiguration import JaxSR

if torch_available:
    from .torch import Torch
