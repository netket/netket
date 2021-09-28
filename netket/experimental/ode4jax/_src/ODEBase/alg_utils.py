
from plum import dispatch

from plum import dispatch

from netket.utils import struct

from .algorithms import AbstractODEAlgorithm
from .controllers import PIController


##
def default_controller(alg, cache, qoldinit=None):
    #if ispredictive(alg): PredictiveController
    # if isstandard(alg): IController
    beta1, beta2 = _digest_beta1_beta2(alg, cache)
    return PIController(beta1, beta2)


def _digest_beta1_beta2(alg, cache):
  beta2 = beta2_default(alg)
  beta1 = beta1_default(alg, beta2)

  # should use rational type...
  #return convert(QT, beta1)::QT, convert(QT, beta2)::QT
  return beta1, beta2


@dispatch
def beta2_default(alg: AbstractODEAlgorithm):
    return 2/(5*alg.alg_order) if alg.is_adaptive else 0


@dispatch
def beta1_default(alg: AbstractODEAlgorithm, beta2):
    return 7/(10*alg.alg_order) if alg.is_adaptive else 0

@dispatch
def gamma_default(alg: AbstractODEAlgorithm, beta2):
    return 9/10 if alg.is_adaptive else 0

