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

from .base import AbstractController

# Standard integral (I) step size controller
class IController(AbstractController):
	"""
	    IController()
	The standard (integral) controller is the most basic step size controller.
	This controller is usually the first one introduced in numerical analysis classes
	but should only be used rarely in practice because of efficiency problems for
	many problems/algorithms.
	Construct an integral (I) step size controller adapting the time step
	based on the formula
	```
	Δtₙ₊₁ = εₙ₊₁^(1/k) * Δtₙ
	```
	where `k = get_current_adaptive_order(alg, integrator.cache) + 1` and `εᵢ` is the
	inverse of the error estimate `integrator.EEst` scaled by the tolerance
	(Hairer, Nørsett, Wanner, 2008, Section II.4).
	The step size factor is multiplied by the safety factor `gamma` and clipped to
	the interval `[qmin, qmax]`.
	A step will be accepted whenever the estimated error `integrator.EEst` is
	less than or equal to unity. Otherwise, the step is rejected and re-tried with
	the predicted step size.
	## References
	- Hairer, Nørsett, Wanner (2008)
	  Solving Ordinary Differential Equations I Nonstiff Problems
	  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
	"""

	@dispatch
	def stepsize_controller(self, integrator, alg):
		"""

		"""
		qmin = integrator.opts.qmin
		qmax = integrator.opts.qmax
		gamma = integrator.opts.gamma

		EEst = integrator.EEst

	    expo = 1 / (get_current_adaptive_order(alg, integrator.cache) + 1)
	    qtmp = EEst**expo  / gamma
	    q = jnp.maximum(1/qmax, jnp.minimum(1/qmin, qtmp))
	    integrator.qold = integrator.dt / q
	    return q

	@dispatch
	def accept_step_controller(self, integrator)
		""" 
		Checks whever the controller should accept a step based on the current
		error estimate
		"""
		return integrator.EEst <= 1

	@dispatch
	def step_accept_controller(self, integrator, alg, q):
		qsteady_min = integrator.opts.qsteady_min
		qsteady_max = integrator.opts.qsteady_max

		isok = (qsteady_min <= q) & (q <= qsteady_max)
		q = jnp.where(isok, jnp.ones_like(q), q)
		return integrator.dt/q 

	@dispatch
	def step_reject_controller(self, integrator, alg, q):
		integrator.dt = integrator.qold


# Standard integral (I) step size controller
@struct.dataclass
class PIController(AbstractController):
	"""
	    PIController(beta1, beta2)
	The proportional-integral (PI) controller is a widespread step size controller
	with improved stability properties compared to the [`IController`](@ref).
	This controller is the default for most algorithms in OrdinaryDiffEq.jl.
	Construct a PI step size controller adapting the time step based on the formula
	```
	Δtₙ₊₁ = εₙ₊₁^β₁ * εₙ^β₂ * Δtₙ
	```
	where `εᵢ` are inverses of the error estimates scaled by the tolerance
	(Hairer, Nørsett, Wanner, 2010, Section IV.2).
	The step size factor is multiplied by the safety factor `gamma` and clipped to
	the interval `[qmin, qmax]`.
	A step will be accepted whenever the estimated error `integrator.EEst` is
	less than or equal to unity. Otherwise, the step is rejected and re-tried with
	the predicted step size.
	!!! note
	    The coefficients `beta1, beta2` are not scaled by the order of the method,
	    in contrast to the [`PIDController`](@ref). For the `PIController`, this
	    scaling by the order must be done when the controller is constructed.
	## References
	- Hairer, Nørsett, Wanner (2010)
	  Solving Ordinary Differential Equations II Stiff and Differential-Algebraic Problems
	  [DOI: 10.1007/978-3-642-05221-7](https://doi.org/10.1007/978-3-642-05221-7)
	- Hairer, Nørsett, Wanner (2008)
	  Solving Ordinary Differential Equations I Nonstiff Problems
	  [DOI: 10.1007/978-3-540-78862-1](https://doi.org/10.1007/978-3-540-78862-1)
	"""

	beta1 : float 
	beta2 : float

	@dispatch
	def stepsize_controller(self, integrator, alg):
		"""

		"""
		qmin = integrator.opts.qmin
		qmax = integrator.opts.qmax
		gamma = integrator.opts.gamma
		qold = integrator.opts.qold

		EEst = integrator.EEst

		q11 = EEst ** self.beta1
		q = q11 / qold ** self.beta2
		integrator.q11 = q11
	    q = jnp.maximum(1/qmax, jnp.minimum(1/qmin, q/tmp))
	    return q

	@dispatch
	def accept_step_controller(self, integrator)
		""" 
		Checks whever the controller should accept a step based on the current
		error estimate
		"""
		return integrator.EEst <= 1

	@dispatch
	def step_accept_controller(self, integrator, alg, q):
		qsteady_min = integrator.opts.qsteady_min
		qsteady_max = integrator.opts.qsteady_max
		qoldinit = integrator.opts.qoldinit
		EEst = integrator.EEst

		isok = (qsteady_min <= q) & (q <= qsteady_max)
		q = jnp.where(isok, jnp.ones_like(q), q)
		integrator.qold = jnp.maximum(EEst, qoldinit)
		return integrator.dt/q 

	@dispatch
	def step_reject_controller(self, integrator, alg, q):
		qmin = integrator.opts.qmin
		gamma = integrator.opts.gamma
		q11 = integrator.opts.q11

		integrator.dt = integrator.dt / jnp.minimum(1/qmin, q11/gamma)

