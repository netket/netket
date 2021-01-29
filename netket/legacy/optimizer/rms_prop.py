from functools import singledispatch
from . import numpy


@singledispatch
def RmsProp(machine, learning_rate=0.001, beta=0.9, epscut=1.0e-7):
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
    return numpy.RmsProp(learning_rate, beta, epscut)
