from functools import singledispatch
from . import numpy


@singledispatch
def AmsGrad(machine, learning_rate=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7):
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

    return numpy.AmsGrad(learning_rate=0.001, beta1=0.9, beta2=0.999, epscut=1.0e-7)
