from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class AdaDelta(AbstractOptimizer):
    r"""AdaDelta Optimizer.
        Like RMSProp, `AdaDelta <http://arxiv.org/abs/1212.5701>`_ corrects the
        monotonic decay of learning rates associated with AdaGrad,
        while additionally eliminating the need to choose a global
        learning rate :math:`\eta`. The NetKet naming convention of
        the parameters strictly follows the one introduced in the original paper;
        here :math:`E[g^2]` is equivalent to the vector :math:`\mathbf{s}` from RMSProp.
        :math:`E[g^2]` and :math:`E[\Delta x^2]` are initialized as zero vectors.

        .. math:: E[g^2]^\prime_k &= \rho E[g^2] + (1-\rho)G_k(\mathbf{p})^2\\
                  \Delta p_k &= - \frac{\sqrt{E[\Delta x^2]+\epsilon}}{\sqrt{E[g^2]+ \epsilon}}G_k(\mathbf{p})\\
                  E[\Delta x^2]^\prime_k &= \rho E[\Delta x^2] + (1-\rho)\Delta p_k^2\\
                  p^\prime_k &= p_k + \Delta p_k
    """

    def __init__(self, rho=0.95, epscut=1.0e-7, l2reg=0):
        r"""
        Constructs a new ``AdaDelta`` optimizer.

        Args:
           rho: Exponential decay rate, in [0,1].
           epscut: Small :math:`\epsilon` cutoff.
           l2reg (float): The amount of L2 regularization.

        Examples:
           Simple AdaDelta optimizer.

           >>> from netket.optimizer import AdaDelta
           >>> op = AdaDelta()
        """

        self._rho = rho
        self._epscut = epscut
        self._l2reg = l2reg

        if epscut <= 0:
            raise ValueError("Invalid epsilon cutoff.")
        if l2reg < 0:
            raise ValueError("Invalid L2 regularization.")
        if rho < 0 or rho > 1:
            raise ValueError("Invalid beta.")

        self._Eg2 = None
        self._Edx2 = None

    def update(self, grad, pars):
        if self._Eg2 is None:
            self._Eg2 = _np.zeros(pars.shape[0])
            self._Edx2 = _np.zeros(pars.shape[0])

        self._Eg2 = self._rho * self._Eg2 + (1.0 - self._rho) * _np.absolute(grad) ** 2

        Dx = _np.sqrt(self._Edx2 + self._epscut) * grad
        Dx /= _np.sqrt(self._Eg2 + self._epscut)

        pars -= Dx

        self._Edx2 = self._rho * self._Edx2 + (1.0 - self._rho) * _np.absolute(Dx) ** 2

        return pars

    def reset(self):
        if self._mt is not None:
            self._Eg2.fill(0.0)
            self._Edx2.fill(0.0)

    def __repr__(self):
        rep = "AdaDelta optimizer with these parameters :"
        rep += "\nRho = " + str(self._rho)
        rep += "\nepscut = " + str(self._epscut)
        return rep
