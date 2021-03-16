from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class Momentum(AbstractOptimizer):
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
    """

    def __init__(self, learning_rate, beta=0.9, l2reg=0):
        r"""
        Constructs a new ``Momentum`` optimizer.

        Args:
            learning_rate (float): The learning rate :math:`\eta`
            beta (float): Momentum exponential decay rate, should be in [0,1].
            l2reg (float): The amount of L2 regularization.

        Examples:
            Momentum optimizer.

            >>> from netket.optimizer import Momentum
            >>> op = Momentum(learning_rate=0.01)
        """
        self._learning_rate = learning_rate
        self._l2reg = l2reg
        self._beta = beta
        self._eta = learning_rate
        self._mt = None

        if learning_rate <= 0:
            raise ValueError("Invalid learning rate.")
        if l2reg < 0:
            raise ValueError("Invalid L2 regularization.")
        if beta < 0 or beta > 1:
            raise ValueError("Invalid beta.")

    def update(self, grad, pars):
        if self._mt is None:
            self._mt = _np.zeros(pars.shape[0], dtype=pars.dtype)

        self._mt = self._beta * self._mt + (1.0 - self._beta) * grad
        pars -= self._eta * (self._mt + self._l2reg * pars)

        return pars

    def reset(self):
        if self._mt is not None:
            self._mt.fill(0.0)

    def __repr__(self):
        rep = "Momentum optimizer with these parameters :"
        rep += "\nLearning Rate = " + str(self._learning_rate)
        rep += "\nL2 Regularization = " + str(self._l2reg)
        rep += "\nBeta = " + str(self._beta)
        return rep
