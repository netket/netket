from typing import Callable, Optional, Union, Tuple, Any
from functools import partial

import jax
import flax
from jax import numpy as jnp
from flax import struct

Ndarray = Any


@struct.dataclass
class SR:
    """
    Base class holding the parameters for the way to find the solution to
    (S + 1*diag_shift) / F.
    """

    diag_shift: float = 0.01
