from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class RmsProp(AbstractOptimizer):
    r"""RMSProp optimizer.

    RMSProp is a well-known update algorithm proposed by Geoff Hinton
    in his Neural Networks course notes `Neural Networks course notes
    <http://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.
    It corrects the problem with AdaGrad by using an exponentially weighted
    moving average over past squared gradients instead of a cumulative sum.
    After initializing the vector :math:`\mathbf{s}` to zero, :math:`s_k` and t
    he parameters :math:`p_k` are updated as

    .. math:: s^\prime_k = \beta s_k + (1-\beta) G_k(\mathbf{p})^2 \\
              p^\prime_k = p_k - \frac{\eta}{\sqrt{s_k}+\epsilon} G_k(\mathbf{p})

    """

    def __init__(self, learning_rate=0.001, beta=0.9, epscut=1.0e-7):
        r"""
        Constructs a new ``RmsProp`` optimizer.

        Args:
            learning_rate: The learning rate :math:`\eta`
            beta: Exponential decay rate.
            epscut: Small cutoff value.

        Examples:
            RmsProp optimizer.

            >>> from netket.optimizer import RmsProp
            >>> op = RmsProp(learning_rate=0.02)
        """

        if epscut <= 0:
            raise ValueError("Invalid epsilon cutoff.")
        if learning_rate < 0:
            raise ValueError("Invalid learning rate.")
        if beta < 0 or beta > 1:
            raise ValueError("Invalid beta.")

        self._eta = learning_rate
        self._beta = beta
        self._epscut = epscut

        self._mt = None

    def update(self, grad, pars):
        if self._mt is None:
            self._mt = _np.zeros(pars.shape[0])

        self._mt = self._beta * self._mt + (1.0 - self._beta) * _np.abs(grad) ** 2

        pars -= self._eta * grad / _np.sqrt(self._mt + self._epscut)

        return pars

    def reset(self):
        if self._mt is not None:
            self._mt.fill(0.0)

    def __repr__(self):
        rep = "AmsGrad optimizer with these parameters :"
        rep += "\nLearning rate = " + str(self._eta)
        rep += "\nbeta = " + str(self._beta)
        rep += "\nepscut = " + str(self._epscut)
        return rep
