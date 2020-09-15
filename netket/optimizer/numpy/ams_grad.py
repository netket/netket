from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class AmsGrad(AbstractOptimizer):
    r"""AmsGrad Optimizer.
        In some cases, adaptive learning rate methods such as AdaMax fail
        to converge to the optimal solution because of the exponential
        moving average over past gradients. To address this problem,
        Sashank J. Reddi, Satyen Kale and Sanjiv Kumar proposed the
        AmsGrad [update algorithm](https://openreview.net/forum?id=ryQu7f-RZ).
        The update rule for :math:`\mathbf{v}` (equivalent to :math:`E[g^2]` in AdaDelta
        and :math:`\mathbf{s}` in RMSProp) is modified such that :math:`v^\prime_k \geq v_k`
        is guaranteed, giving the algorithm a "long-term memory" of past gradients.
        The vectors :math:`\mathbf{m}` and :math:`\mathbf{v}` are initialized to zero, and
        are updated with the parameters :math:`\mathbf{p}`:

        .. math:: m^\prime_k &= \beta_1 m_k + (1-\beta_1)G_k(\mathbf{p})\\
                  v^\prime_k &= \beta_2 v_k + (1-\beta_2)G_k(\mathbf{p})^2\\
                  v^\prime_k &= \mathrm{Max}(v^\prime_k, v_k)\\
                  p^\prime_k &= p_k - \frac{\eta}{\sqrt{v^\prime_k}+\epsilon}m^\prime_k

    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7):
        r"""
        Constructs a new ``AmsGrad`` optimizer.

        Args:
            learning_rate: The learning rate $\eta$.
            beta1: First exponential decay rate.
            beta2: Second exponential decay rate.
            epscut: Small epsilon cutoff.

        Examples:
            Simple AmsGrad optimizer.

            >>> from netket.optimizer import AmsGrad
            >>> op = AmsGrad()
        """

        if epscut <= 0:
            raise ValueError("Invalid epsilon cutoff.")
        if learning_rate < 0:
            raise ValueError("Invalid learning rate.")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("Invalid beta1.")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("Invalid beta1.")

        self._eta = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epscut = epscut

        self._mt = None
        self._vt = None
        self._niter = 0

    def update(self, grad, pars):
        if self._mt is None:
            self._mt = _np.zeros(pars.shape[0])
            self._vt = _np.zeros(pars.shape[0])

        self._mt = self._beta1 * self._mt + (1.0 - self._beta1) * grad

        self._vt = _np.maximum(
            self._vt, self._beta2 * self._vt + (1 - self._beta2) * _np.abs(grad) ** 2.0
        )

        pars -= self._eta * self._mt / _np.sqrt(self._vt + self._epscut)

        return pars

    def reset(self):
        if self._mt is not None:
            self._mt.fill(0.0)
            self._vt.fill(0.0)

    def __repr__(self):
        rep = "AmsGrad optimizer with these parameters :"
        rep += "\nLearning rate = " + str(self._eta)
        rep += "\nbeta1 = " + str(self._beta1)
        rep += "\nbeta2 = " + str(self._beta2)
        rep += "\nepscut = " + str(self._epscut)
        return rep
