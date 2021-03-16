from functools import singledispatch
from . import numpy


@singledispatch
def AdaGrad(machine, learning_rate=0.001, epscut=1.0e-7):
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


        Args:
           learning_rate: Learning rate :math:`\eta`.
           epscut: Small :math:`\epsilon` cutoff.

        Examples:
           Simple AdaGrad optimizer.

           >>> from netket.optimizer import AdaGrad
           >>> op = AdaGrad()
        """

    return numpy.AdaGrad(learning_rate, epscut)
