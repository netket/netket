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

from typing import Callable

from plum import dispatch

from builtins import RuntimeError, next
from functools import partial
from typing import Callable, Optional, Tuple, Type

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import Array, PyTree

dtype = jnp.float64

@struct.dataclass(_frozen=False)
class AbstractIntegrator:

	def step(self):
		"""
		Performs one step of integration
		"""
		raise NotImplementedError("Integrator stepping is not implemented")

	def du(self):
		"""
		Return the derivative at current t
		"""
		raise NotImplementedError

	def proposed_dt(self):
		"""
		gets the proposed dt for the next timestep
		"""
		raise NotImplementedError

	def set_proposde_dt(self):
		"""
		returns a new integrator with the proposed value of dt set
		"""
		raise NotImplementedError

	def set_savevalues(self, force_save):
		"""
		    savevalues!(integrator::DEIntegrator,
		      force_save=false) -> Tuple{Bool, Bool}
		Try to save the state and time variables at the current time point, or the
		`saveat` point by using interpolation when appropriate. It returns a tuple that
		is `(saved, savedexactly)`. If `savevalues!` saved value, then `saved` is true,
		and if `savevalues!` saved at the current time point, then `savedexactly` is
		true.
		The saving priority/order is as follows:
		  - `save_on`
		    - `saveat`
		    - `force_save`
		    - `save_everystep`
		"""
		raise NotImplementedError

	def add_stop(self, t):
		raise NotImplementedError

	def add_saveat(self, t):
		raise NotImplementedError

	def set_abstol(self, t):
		raise NotImplementedError

	def set_reltol(self, t):
		raise NotImplementedError

	def reinit(self, *args):
		"""
		resets the integrator
		"""