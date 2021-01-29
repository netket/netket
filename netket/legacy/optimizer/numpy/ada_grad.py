from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class AdaGrad(AbstractOptimizer):
    r"""AdaGrad Optimizer.
        In many cases, in Sgd the learning rate :math`\eta` should
        decay as a function of training iteration to prevent overshooting
        as the optimum is approached. AdaGrad is an adaptive learning
        rate algorithm that automatically scales the learning rate with a sum
        over past gradients. The vector :math:`\mathbf{g}` is initialized to zero.
        Given a stochastic estimate of the gradient of the cost function :math:`G(\mathbf{p})`,
        the updates for :math:`g_k` and the parameter :math:`p_k` are


        .. math:: g^\prime_k &= g_k + G_k(\mathbf{p})^2\\
                  p^\prime_k &= p_k - \frac{\eta}{\sqrt{g_k + \epsilon}}G_k(\mathbf{p})

        AdaGrad has been shown to perform particularly well when
        the gradients are sparse, but the learning rate may become too small
        after many updates because the sum over the squares of past gradients is cumulative.
    """

    def __init__(self, learning_rate=0.001, epscut=1.0e-7):
        r"""
        Constructs a new ``AdaGrad`` optimizer.

        Args:
            learning_rate: Learning rate :math:`\eta`.
            epscut: Small :math:`\epsilon` cutoff.

        Examples:
            Simple AdaGrad optimizer.

            >>> from netket.optimizer import AdaGrad
            >>> op = AdaGrad()
        """

        self._eta = learning_rate
        self._epscut = epscut

        if epscut <= 0:
            raise ValueError("Invalid epsilon cutoff.")
        if learning_rate < 0:
            raise ValueError("Invalid learning rate.")

        self._Gt = None

    def update(self, grad, pars):
        if self._Gt is None:
            self._Gt = _np.zeros(pars.shape[0])

        self._Gt += _np.abs(grad) ** 2

        pars -= self._eta * grad / _np.sqrt(self._Gt + self._epscut)

        return pars

    def reset(self):
        if self._mt is not None:
            self._Gt.fill(0.0)

    def __repr__(self):
        rep = "AdaGrad optimizer with these parameters :"
        rep += "\nLearning Rate = " + str(self._eta)
        rep += "\nepscut = " + str(self._epscut)
        return rep
