from .metropolis_hastings import MetropolisHastings
from .metropolis_exchange import ExchangeKernel as NumpyExchangeKernel
from functools import singledispatch


@singledispatch
def ExchangeKernel(machine, dmax):
    return NumpyExchangeKernel(machine.hilbert, dmax)
