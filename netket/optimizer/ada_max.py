from functools import singledispatch
from . import numpy


@singledispatch
def AdaMax(machine, alpha=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7):
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

    return numpy.AdaMax(alpha, beta1, beta2, epscut)
