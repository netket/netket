
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

		u_t, err_t = tableau.step_with_error(integrator.f, integrator.t, integrator.dt, integrator.u)
		t = integrator.t + integrator.dt

		integrator.u = u_t
		# integrator.t = t
		# todo update error
	else:
		u_t = tableau.step(integrator.f, integrator.t, integrator.dt, integrator.u)
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


##
def default_controller(alg, cache, qoldinit):
	#if ispredictive(alg): PredictiveController
	# if isstandard(alg): IController
	beta1, beta2 = _digest_beta1_beta2(alg, cache)
	return PIController(beta1, beta2)


#def _digest_beta1_beta2(alg, cache, QT, _beta1, _beta2):
#  if typeof(alg) <: OrdinaryDiffEqCompositeAlgorithm
#    beta2 = _beta2 === nothing ? _composite_beta2_default(alg.algs, cache.current, QT) : _beta2
#    beta1 = _beta1 === nothing ? _composite_beta1_default(alg.algs, cache.current, QT, beta2) : _beta1
#  else
#    beta2 = _beta2 === nothing ? beta2_default(alg) : _beta2
#    beta1 = _beta1 === nothing ? beta1_default(alg,beta2) : _beta1
#  end
#  return convert(QT, beta1)::QT, convert(QT, beta2)::QT
#