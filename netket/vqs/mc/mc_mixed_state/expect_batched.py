from functools import partial
from typing import Callable

import numpy as np

import jax
from jax import numpy as jnp

from netket import jax as nkjax
from netket import config
from netket.stats import Stats
from netket.utils.types import PyTree
from netket.utils.dispatch import dispatch

from netket.operator import (
    AbstractSuperOperator,
    DiscreteOperator,
    Squared,
)

from netket.vqs.mc import get_fun

from .state import MCMixedState

# Dispatches to select what expect-kernel to use
@dispatch
def get_fun(vstate: MCMixedState, Ô: Squared[AbstractSuperOperator], batch_size: int):
    return kernels.local_value_squared_kernel_batched

@dispatch
def get_fun(vstate: MCMixedState, Ô: DiscreteOperator, batch_size: int):
    return kernels.local_value_op_op_cost_batched

