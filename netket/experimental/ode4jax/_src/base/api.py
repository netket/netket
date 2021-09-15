
from plum import dispatch

from .problem import AbstractProblem
from .solver import AbstractAlgorithm
from .integrator import AbstractIntegrator

@dispatch
def solve(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
	return _solve(problem, solver, *args, **kwargs)

def init(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
	# use to preprocess arguments
	integrator = _init(problem, solver, *args, **kwargs)
	_initialize(integrator)
	return integrator

def step(integrator: AbstractIntegrator):
	return _step(integrator)

# extension points
@dispatch.abstract
def _solve(problem, solver, *args, **kwargs):
	pass

@dispatch.abstract
def _init(problem: AbstractProblem, solver: AbstractAlgorithm, *args, **kwargs):
	pass

@dispatch.abstract
def _initialize(integrator: AbstractIntegrator):
	pass

@dispatch.abstract
def _step(integrator: AbstractIntegrator):
	pass

