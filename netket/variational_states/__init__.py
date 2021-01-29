from .base import VariationalState, VariationalMixedState
from .classical import ClassicalVariationalState
from .mixed_classical import ClassicalVariationalMixedState

ClassicalVariationalPureState = ClassicalVariationalState

from netket.utils import _hide_submodules

_hide_submodules(__name__)
