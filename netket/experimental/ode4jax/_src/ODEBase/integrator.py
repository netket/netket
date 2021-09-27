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

from ..base import AbstractIntegrator, AbstractSolution
from ..base import alg_cache

from .rk import AbstractODEAlgorithm

from .problem import ODEProblem
from .solution import ODESolution
from .options import DEOptions

from .utils import strong_dtype

@struct.dataclass(_frozen=False)
class ODEIntegrator(AbstractIntegrator):
	solution: AbstractSolution
	u: PyTree
	t: float
	tdir:float 
	tprev: float
	dt: float
	dtcache: float
	dtpropose:float
	iter: int
	alg: AbstractODEAlgorithm
	success_iter:int
	opts: DEOptions
	saveiter:int
	saveiter_dense:int
	f: Callable = struct.field(pytree_node=False)
	force_stepfail : bool 
	error_code : int

	cache: Any

	@property
	def has_tstops(self):
		return self.opts.next_tstop_id < self.opts.tstops.size

	@property
	def first_tstop(self):
		return self.opts.tstops[self.opts.next_tstop_id]



@dispatch
def _init(problem: ODEProblem, alg: AbstractODEAlgorithm, *, abstol=None, reltol=None, dt=None, dtmin=None, dtmax=None, 
	saveat=None, save_start=None, save_end=None, tstops=None, callback=None, controller=None, save_everystep=None,
	maxiters=None, adaptive=None):
	
	tspan = problem.tspan
	tdir = jnp.sign(tspan[-1] - tspan[0])
	t = jnp.array(problem.tspan[0], dtype=float)
	u = problem.u0

	if abstol is None:
		abstol_internal = 1/10**6
	else:
		# TODO staging
		abstol_internal = jnp.real(abstol)

	if reltol is None:
		reltol_internal = 1/10**3
	else:
		# TODO staging
		reltol_internal = jnp.real(reltol)

	if adaptive is None:
		adaptive = alg.is_adaptive

	if maxiters is None:
		#maxiters = 1000000 if adaptive else np.iinfo(np.int32).max
		maxiters = 1000000

	if dtmax is None:
		dtmax = problem.tspan[-1] - problem.tspan[0]

	# convert negative dtmax to positive
	if dtmax > 0 and tdir < 0:
		dtmax = dtmax * tdir

	if dtmin is None:
		dtmin = problem.dtmin(use_end_time=False)

	if dt is None:
		if tstops is not None and len(tstops) > 2:
			# todo 
			pass
		elif adaptive is False:
			raise ValueError("Must specify dt for non-adaptive solvers algorithms.")

		dt = jnp.array(0)

	dt = strong_dtype(dt)
	# concretize dt
	cache = alg_cache(alg, u, reltol_internal)

	#if controller is None:
		#controller = default_controller(alg, cache) 

	if tstops is None:
		if adaptive:
			tstops = jnp.asarray([tspan[1]])
		else:
			tstops = jnp.asarray([tspan[1]])
	elif isinstance(tstops, Number):
		tstops = jnp.linspace(tspan[0], tspan[1], num=tstops)

	if saveat is None:
		saveat = jnp.asarray([])
	elif isinstance(saveat, Number):
		saveat = jnp.linspace(tspan[0], tspan[1], num=saveat)

	if save_everystep is None:
		save_everystep = len(saveat) == 0

	if save_start is None:
		save_start = save_everystep or len(saveat) == 0 or isinstance(saveat, Number)# or problem.tspan[0] in saveat

	if save_end is None:
		save_end = save_everystep or len(saveat) == 0 or isinstance(saveat, Number)# or problem.tspan[1] in saveat

	opts = DEOptions(maxiters=maxiters, adaptive=adaptive, abstol=abstol_internal, reltol=reltol_internal, 
		controller=controller, saveat=jnp.sort(saveat), next_saveat_id=0, tstops=jnp.sort(tstops), save_everystep=save_everystep, 
		save_start=save_start, save_end=save_end, next_tstop_id=0, dtmax=dtmax, dtmin=dtmin)


	n_saved_pts = len(saveat) + save_start + save_end
	solution = ODESolution.make(u, n_saved_pts)

	return ODEIntegrator(solution=solution, u=u, t=t, tprev=t, f=problem.f, alg=alg, tdir=tdir, 
						 success_iter=0, iter=0, opts=opts, saveiter=0, saveiter_dense=0,
						 dt=dt, dtcache=dt, dtpropose=dt, force_stepfail=jnp.asarray(False), 
						 error_code=jnp.array(False, dtype=bool), cache=cache)

@dispatch
def _initialize(integrator: ODEIntegrator):
	if integrator.opts.save_start:
		integrator.solution.set(integrator.saveiter, integrator.t, integrator.u)

		integrator.saveiter += 1
		integrator.saveiter_dense += 1

		cond = integrator.opts.tstops[integrator.opts.next_tstop_id] <= integrator.t
		integrator.opts.next_tstop_id = jax.lax.cond(cond, 
			lambda _ : integrator.opts.next_tstop_id + 1, 
			lambda _: integrator.opts.next_tstop_id,
			None)

		cond = integrator.opts.saveat[integrator.opts.next_saveat_id] <= integrator.t
		integrator.opts.next_saveat_id = jax.lax.cond(cond, 
			lambda _ : integrator.opts.next_saveat_id + 1, 
			lambda _: integrator.opts.next_saveat_id,
			None)

	# initialize callback
	# initialize cache

	# initialize dt
	handle_dt(integrator)

def handle_dt(integrator):
	if integrator.opts.adaptive:
		cond = lambda : jnp.all(integrator.dt == 0) and integrator.adaptive

	#function handle_dt!(integrator)
	#  if iszero(integrator.dt) && integrator.opts.adaptive
	#    auto_dt_reset!(integrator)
	#    if sign(integrator.dt)!=integrator.tdir && !iszero(integrator.dt) && !isnan(integrator.dt)
	#      error("Automatic dt setting has the wrong sign. Exiting. Please report this error.")
	#    end
	#    if isnan(integrator.dt)
	#      if integrator.opts.verbose
	#        @warn("Automatic dt set the starting dt as NaN, causing instability.")
	#      end
	#    end
	#  elseif integrator.opts.adaptive && integrator.dt > zero(integrator.dt) && integrator.tdir < 0
	#    integrator.dt *= integrator.tdir # Allow positive dt, but auto-convert
	#  end
	#end
	pass

				