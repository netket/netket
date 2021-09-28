# Copyright 2021 The NetKet Authors - All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Callable, Tuple
from numbers import Number

from textwrap import dedent

from builtins import RuntimeError, next
import dataclasses
from functools import partial
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp
import numpy as np

from netket.utils import struct
from netket.utils.types import Array, PyTree

dtype = jnp.float64

from ..base import AbstractProblem
from netket.experimental.pytreearray import PyTreeArray1

@struct.dataclass
class ODEProblem(AbstractProblem):
    """
    Represents an Ordinary Differential Equation Problem, defined by the
    function `f`, from t0 to t1 defined in tspan, and with initial
    conditions `u0`.

    Optionally a set of parameters `p` can also be specified.
    """
    f: Callable = struct.field(pytree_node=False)
    tspan: Tuple[float]
    u0: PyTree

    p: Any

    def __pre_init__(self, fun, tspan, u0, *, p=()):
        if len(tspan) != 2:
            raise ValueError("tspan must be a tuple of length 2")

        tspan = (jnp.asarray(tspan[0]), jnp.asarray(tspan[1]))

        if isinstance(u0, np.ndarray) or isinstance(u0, jnp.ndarray):
            u0 = jnp.asarray(u0)
        else:
            u0 = PyTreeArray1(u0)

        if u0.ndim == 0:
            u0 = u0.reshape(1)

        return (fun, tspan, u0, p), {}

    def dtmin(self, *, use_end_time=True):
        #if use_end_time:
        #   dtmin = t1f & t2f ? max(eps(t1), eps(t2)) : max(eps(typeof(t1)), eps(t1))
        dtmin = 1e-16
        return dtmin

    def __repr__(self):
        u0 = self.u0
        if isinstance(u0, jnp.ndarray):
            u0str = f"{u0.dtype}{u0.shape} array with {u0.size} elements"
        else:
            u0str = f"PyTreeArray with {u0.size} elements its leafs"

        return dedent(
        f"""
        ODEProblem:
            tspan: {self.tspan[0]} → {self.tspan[1]}
            u0: {u0str}
        """)[1:]

