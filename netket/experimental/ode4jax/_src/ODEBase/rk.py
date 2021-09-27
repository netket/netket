
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

	@property
	def alg_order(self):
		raise NotImplementedError

@struct.dataclass
class AbstractODETableauAlgorithm(AbstractODEAlgorithm):
	@property
	def tableau(self):
		raise NotImplementedError

	@property
	def alg_order(self):
		return self.tableau.order[0]

	@property
	def n_stages(self):
		return self.tableau.stages

	@property
	def is_adaptive(self):
		return self.tableau.is_adaptive


@struct.dataclass
class AbstractODERKAlgorithm(AbstractODETableauAlgorithm):
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

		u_t, err_t = tableau.step_with_error(integrator)
		t = integrator.t + integrator.dt

		integrator.u = u_t
		# integrator.t = t
		# todo update error
	else:
		u_t = tableau.step(integrator)
		t = integrator.t + integrator.dt

		integrator.u = u_t
		# integrator.t = t

	return integrator


##
@dispatch
def get_current_adaptive_order(alg: AbstractODEAlgorithm, cache):
	pass

@dispatch
def get_current_adaptive_order(alg: AbstractODERKAlgorithm, cache):
	return alg.tableau[0]
