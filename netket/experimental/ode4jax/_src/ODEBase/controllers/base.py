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

from typing import Any, Callable
from numbers import Number

from plum import dispatch

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

from ...base import AbstractIntegrator, AbstractSolution
from ...base import alg_cache

from ..rk import AbstractODEAlgorithm

from ..problem import ODEProblem
from ..solution import ODESolution
from ..options import DEOptions

from ..utils import strong_dtype

@struct.dataclass
class AbstractController:

	@dispatch
	def stepsize_controller(self, integrator, alg):
		"""

		"""
		pass

	@dispatch
	def accept_step_controller(self, integrator)
		""" 
		Checks whever the controller should accept a step based on the current
		error estimate
		"""
		return integrator.EEst <= 1

	@dispatch
	def step_accept_controller(self, integrator, alg, q):
		pass

	@dispatch
	def step_reject_controller(self, integrator, alg, q):
		pass

	@dispatch
	def reset_alg_dependent_opts(self, alg1, alg2):
		pass

	@dispatch
	def reinit(self, integrator):
		pass