
from plum import dispatch

from plum import dispatch

from netket.utils import struct

from ..base import AbstractAlgorithm
from ..base import AbstractIntegrator

from . import tableau

@struct.dataclass
class AbstractODEAlgorithm(AbstractAlgorithm):

	@property
	def is_adaptive(self):
		return False

@struct.dataclass
class AbstractODERKAlgorithm(AbstractODEAlgorithm):
	pass

@struct.dataclass
class Euler(AbstractODERKAlgorithm):
	
	@property
	def tableau(self):
		return tableau.bt_feuler

@struct.dataclass
class RK4(AbstractODERKAlgorithm):
	@property
	def tableau(self):
		return tableau.bt_rk4


@struct.dataclass
class AbstractODERKAlgorithmCache:
	pass

@struct.dataclass
class FixedDtRKAlgorithmCache(AbstractODERKAlgorithmCache):
	pass

@dispatch
def alg_cache(alg : AbstractODERKAlgorithm, u, reltol_internal):
	return FixedDtRKAlgorithmCache()

@dispatch
def perform_step(integrator: AbstractIntegrator, cache: AbstractODERKAlgorithmCache, *, repeat_step=False):
	tableau = integrator.alg.tableau

	if integrator.opts.adaptive:

		u_t = tableau.step(integrator.f, integrator.t, integrator.dt, integrator.u)
		t = integrator.t + integrator.dt

		integrator.u = u_t
		#integrator.t = t
	else:
		u_t, err_t = tableau.step_with_error(integrator.f, integrator.t, integrator.dt, integrator.u)
		t = integrator.t + integrator.dt

		integrator.u = u_t
		#integrator.t = t
		# todo update error

	return integrator

