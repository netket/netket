from netket.legacy.machine import Jax as JaxMachine
from ..metropolis_hastings import MetropolisHastings, MetropolisHastingsPt
from .metropolis_hastings import MetropolisHastings as JaxMetropolisHastings


@MetropolisHastings.register(JaxMachine)
def _JaxMetropolisHastings(machine, kernel, n_chains=16, sweep_size=None, rng_key=None):
    return JaxMetropolisHastings(machine, kernel, n_chains, sweep_size, rng_key)


@MetropolisHastingsPt.register(JaxMachine)
def _JaxMetropolisHastingsPt(
    machine, kernel, n_replicas=32, sweep_size=None, rng_key=None
):
    raise NotImplementedError("Parallel tempering samplers not yet implemented in Jax")


# Register Jax kernels here
from .._kernels import _LocalKernel, _ExchangeKernel, _HamiltonianKernel, _CustomKernel
from .local_kernel import _JaxLocalKernel
from .exchange_kernel import _JaxExchangeKernel


@_LocalKernel.register(JaxMachine)
def _Jax_LocalKernel(machine):
    return _JaxLocalKernel(machine.hilbert.local_states, machine.input_size)


@_ExchangeKernel.register(JaxMachine)
def _Jax_ExchangeKernel(machine, clusters):
    return _JaxExchangeKernel(machine.hilbert, clusters)


@_HamiltonianKernel.register(JaxMachine)
def _Jax_HamiltonianKernel(machine, hamiltonian):
    raise NotImplementedError("Kernel not yet implemented in Jax")


@_CustomKernel.register(JaxMachine)
def _Jax_CustomKernel(machine, move_operators, move_weights=None):
    raise NotImplementedError("Kernel not yet implemented in Jax")
