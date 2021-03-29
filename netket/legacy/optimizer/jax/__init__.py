from .wrap import Wrap
import jax
import jax.experimental.optimizers as jaxopt
from .. import AdaDelta, AdaGrad, AdaMax, AmsGrad, Momentum, RmsProp, Sgd

from netket.legacy.machine import Jax as JaxMachine


@AdaDelta.register(JaxMachine)
def _JaxAdaDelta(machine, rho=0.95, epscut=1.0e-7, l2reg=0):
    raise NotImplementedError("Optimizer not implemented yet in Jax")


@AdaGrad.register(JaxMachine)
def _JaxAdaGrad(machine, learning_rate=0.001, epscut=1.0e-7):
    return Wrap(machine, jaxopt.adagrad(learning_rate))


@AdaMax.register(JaxMachine)
def _JaxAdaMax(machine, alpha=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7):
    return Wrap(machine, jaxopt.adamax(alpha, beta1, beta2, epscut))


@AmsGrad.register(JaxMachine)
def _JaxAdaDelta(machine, rho=0.95, epscut=1.0e-7, l2reg=0):
    raise NotImplementedError("Optimizer not implemented yet in Jax")


@Momentum.register(JaxMachine)
def _JaxMomentum(machine, learning_rate, beta=0.9, l2reg=0):
    return Wrap(machine, jaxopt.momentum(learning_rate, beta))


@RmsProp.register(JaxMachine)
def _JaxRmsProp(machine, learning_rate=0.001, beta=0.9, epscut=1.0e-7):
    return Wrap(machine, jaxopt.rmsprop(learning_rate, beta, epscut))


@Sgd.register(JaxMachine)
def _JaxSgd(machine, learning_rate, l2reg=0, decay_factor=1.0):
    return Wrap(machine, jaxopt.sgd(learning_rate))


# SR
from .. import SR as _SR
from .stochastic_reconfiguration import SR


@_SR.register(JaxMachine)
def _JaxSR(
    machine,
    lsq_solver=None,
    diag_shift=0.01,
    use_iterative=True,
    svd_threshold=None,
    sparse_tol=None,
    sparse_maxiter=None,
    onthefly=True,
):
    return SR(
        machine,
        lsq_solver,
        diag_shift,
        use_iterative,
        svd_threshold,
        sparse_tol,
        sparse_maxiter,
        onthefly,
    )
