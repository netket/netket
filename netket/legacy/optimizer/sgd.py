from functools import singledispatch
from . import numpy


@singledispatch
def Sgd(machine, learning_rate, l2reg=0, decay_factor=1.0):
    r"""Stochastic Gradient Descent Optimizer.
    The `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
    is one of the most popular optimizers in machine learning applications.
    Given a stochastic estimate of the gradient of the cost function (:math:`G(\mathbf{p})`),
    it performs the update:

    .. math:: p^\prime_k = p_k -\eta G_k(\mathbf{p}),

    where :math:`\eta` is the so-called learning rate.
    NetKet also implements two extensions to the simple SGD,
    the first one is :math:`L_2` regularization,
    and the second one is the possibility to set a decay
    factor :math:`\gamma \leq 1` for the learning rate, such that
    at iteration :math:`n` the learning rate is :math:`\eta \gamma^n`.

    Args:
       machine (netket.AbstractMachine): The machine whose parameters are being optimized.
       learning_rate (float): The learning rate :math:`\eta`.
       l2_reg (float): The amount of :math:`L_2` regularization.
       decay_factor (float): The decay factor :math:`\gamma`.

    Examples:
       Simple SGD optimizer.

       >>> from netket.optimizer import Sgd
       >>> op = Sgd(learning_rate=0.05)
    """
    return numpy.Sgd(learning_rate, l2reg, decay_factor)
