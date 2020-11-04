from functools import singledispatch
from . import numpy


@singledispatch
def Momentum(machine, learning_rate, beta=0.9, l2reg=0):
    r"""Momentum-based Optimizer.
        The momentum update incorporates an exponentially weighted moving average
        over previous gradients to speed up descent
        `Qian, N. (1999) <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf>`_.
        The momentum vector :math:`\mathbf{m}` is initialized to zero.
        Given a stochastic estimate of the gradient of the cost function
        :math:`G(\mathbf{p})`, the updates for the parameter :math:`p_k` and
        corresponding component of the momentum :math:`m_k` are

        .. math:: m^\prime_k &= \beta m_k + (1-\beta)G_k(\mathbf{p})\\
        p^\prime_k &= \eta m^\prime_k


        Args:
           learning_rate (float): The learning rate :math:`\eta`
           beta (float): Momentum exponential decay rate, should be in [0,1].
           l2reg (float): The amount of L2 regularization.

        Examples:
           Momentum optimizer.

           >>> from netket.optimizer import Momentum
           >>> op = Momentum(learning_rate=0.01)
    """
    return numpy.Momentum(learning_rate, beta, l2reg)
