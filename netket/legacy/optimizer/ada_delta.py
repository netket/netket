from functools import singledispatch
from . import numpy


@singledispatch
def AdaDelta(machine, rho=0.95, epscut=1.0e-7, l2reg=0):
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


        Args:
           rho: Exponential decay rate, in [0,1].
           epscut: Small :math:`\epsilon` cutoff.
           l2reg (float): The amount of L2 regularization.

        Examples:
           Simple AdaDelta optimizer.

           >>> from netket.optimizer import AdaDelta
           >>> op = AdaDelta()
    """
    return numpy.AdaDelta(rho, epscut, l2reg)
