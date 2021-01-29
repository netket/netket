from ..abstract_optimizer import AbstractOptimizer
import numpy as _np


class AdaMax(AbstractOptimizer):
    r"""AdaMax Optimizer.

    AdaMax is an adaptive stochastic gradient descent method,
    and a variant of `Adam <https://arxiv.org/pdf/1412.6980.pdf>`_ based on the infinity norm.
    In contrast to the SGD, AdaMax offers the important advantage of being much
    less sensitive to the choice of the hyper-parameters (for example, the learning rate).

    Given a stochastic estimate of the gradient of the cost function :math:`G(\mathbf{p})`,
    AdaMax performs an update:

    .. math:: p^{\prime}_k = p_k + \mathcal{S}_k,

    where :math:`\mathcal{S}_k` implicitly depends on all the history of the optimization up to the current point.
    The NetKet naming convention of the parameters strictly follows the one introduced by the authors of AdaMax.
    For an in-depth description of this method, please refer to
    `Kingma, D., & Ba, J. (2015). Adam: a method for stochastic optimization <https://arxiv.org/pdf/1412.6980.pdf>`_
    (Algorithm 2 therein).
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7):
        r"""
        Constructs a new ``AdaMax`` optimizer.

        Args:
            alpha: The step size.
            beta1: First exponential decay rate.
            beta2: Second exponential decay rate.
            epscut: Small epsilon cutoff.

        Examples:
            Simple AdaMax optimizer.

            >>> from netket.optimizer import AdaMax
            >>> op = AdaMax()
        """

        if epscut <= 0:
            raise ValueError("Invalid epsilon cutoff.")
        if alpha < 0:
            raise ValueError("Invalid alpha.")
        if beta1 < 0 or beta1 > 1:
            raise ValueError("Invalid beta1.")
        if beta2 < 0 or beta2 > 1:
            raise ValueError("Invalid beta1.")

        self._alpha = alpha
        self._beta1 = beta1
        self._beta2 = beta2
        self._epscut = epscut

        self._mt = None
        self._ut = None
        self._niter = 0

    def update(self, grad, pars):
        if self._mt is None:
            self._mt = _np.zeros(pars.shape[0])
            self._ut = _np.zeros(pars.shape[0])

        self._mt = self._beta1 * self._mt + (1.0 - self._beta1) * grad

        self._ut = _np.maximum(
            _np.maximum(_np.abs(grad), self._beta2 * self._ut), self._epscut
        )

        self._niter += 1

        eta = self._alpha / (1.0 - self._beta1 ** self._niter)

        pars -= eta * self._mt / self._ut

        return pars

    def reset(self):
        if self._mt is not None:
            self._mt.fill(0.0)
            self._ut.fill(0.0)
            self._niter = 0

    def __repr__(self):
        rep = "AdaMax optimizer with these parameters :"
        rep += "\nStep size alpha = " + str(self._alpha)
        rep += "\nbeta1 = " + str(self._beta1)
        rep += "\nbeta2 = " + str(self._beta2)
        rep += "\nepscut = " + str(self._epscut)
        return rep
