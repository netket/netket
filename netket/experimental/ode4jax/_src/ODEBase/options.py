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

from typing import Callable, Any

from builtins import RuntimeError, next
import dataclasses
from functools import partial
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array, PyTree

dtype = jnp.float64

from ..base import AbstractProblem

@struct.dataclass(_frozen=False)
class DEOptions:
	abstol: float
	reltol: float

	tstops : Any
	next_tstop_id : int 

	saveat : Any
	next_saveat_id : int
	controller: Any

	dtmin : float
	dtmax : float

	save_start: bool = struct.field(pytree_node=False, default=False)
	save_end: bool = struct.field(pytree_node=False, default=False)
	maxiters: int = struct.field(pytree_node=False, default=0)
	save_everystep : bool = struct.field(pytree_node=False, default=False)
	adaptive : bool = struct.field(pytree_node=False, default=True)


