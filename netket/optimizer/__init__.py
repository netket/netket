# Copyright 2021 The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from . import qgt, solver

from .sr import SR

from .linear_operator import LinearOperator
from .preconditioner import (
    LinearPreconditioner,
    PreconditionerT,
    identity_preconditioner,
)

## Optimisers

from netket.utils import _hide_submodules

from ._optax import split_complex


def Sgd(learning_rate: float):
    r"""Stochastic Gradient Descent Optimizer.
    The `Stochastic Gradient Descent <https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_
    is one of the most popular optimizers in machine learning applications.
    Given a stochastic estimate of the gradient of the cost function (:math:`G(\mathbf{p})`),
    it performs the update:

    .. math:: p^\prime_k = p_k -\eta G_k(\mathbf{p}),

    where :math:`\eta` is the so-called learning rate.
    NetKet also implements two extensions to the simple SGD,
    the first one is :math:`L_2` regularization,
    and the second one is the possibility to set a decay
    factor :math:`\gamma \leq 1` for the learning rate, such that
    at iteration :math:`n` the learning rate is :math:`\eta \gamma^n`.

    Args:
       learning_rate: The learning rate :math:`\eta`.

    Examples:
       Simple SGD optimizer.

       >>> from netket.optimizer import Sgd
       >>> op = Sgd(learning_rate=0.05)
    """
    from optax import sgd

    return sgd(learning_rate)


def Momentum(learning_rate: float, beta: float = 0.9, nesterov: bool = False):
    r"""Momentum-based Optimizer.
        The momentum update incorporates an exponentially weighted moving average
        over previous gradients to speed up descent
        `Qian, N. (1999) <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.57.5612&rep=rep1&type=pdf>`_.
        The momentum vector :math:`\mathbf{m}` is initialized to zero.
        Given a stochastic estimate of the gradient of the cost function
        :math:`G(\mathbf{p})`, the updates for the parameter :math:`p_k` and
        corresponding component of the momentum :math:`m_k` are

        .. math:: m^\prime_k &= \beta m_k + (1-\beta)G_k(\mathbf{p})\\
        p^\prime_k &= \eta m^\prime_k


        Args:
           learning_rate: The learning rate :math:`\eta`
           beta: Momentum exponential decay rate, should be in [0,1].
           nesterov: Flag to use nesterov momentum correction

        Examples:
           Momentum optimizer.

           >>> from netket.optimizer import Momentum
           >>> op = Momentum(learning_rate=0.01)
    """
    from optax import sgd

    return sgd(learning_rate, momentum=beta, nesterov=nesterov)


def AdaGrad(
    learning_rate: float = 0.001,
    epscut: float = 1.0e-7,
    initial_accumulator_value: float = 0.1,
):
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
           initial_accumulator_value: initial value of the accumulator

        Examples:
           Simple AdaGrad optimizer.

           >>> from netket.optimizer import AdaGrad
           >>> op = AdaGrad()
        """
    from optax import adagrad

    return adagrad(
        learning_rate, eps=epscut, initial_accumulator_value=initial_accumulator_value
    )


def Adam(learning_rate: float = 0.001, b1: float = 0.9, b2: float = 0.999, eps=1e-08):
    r"""Adam Optimizer.

    Args:
        learning_rate: Learning rate :math:`\eta`.
        b1: Decay rate for the exponentially weighted average of grads.
        b2: Decay rate for the exponentially weighted average of squared norm of grads.
        eps: Term added to the denominator to improve numerical stability.
    """
    from ._optax import adam

    return adam(learning_rate, b1=b1, b2=b2, eps=eps)


def RmsProp(
    learning_rate: float = 0.001,
    beta: float = 0.9,
    epscut: float = 1.0e-7,
    centered: bool = False,
):
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
       centered: whever to center the moving average.

    Examples:
       RmsProp optimizer.

       >>> from netket.optimizer import RmsProp
       >>> op = RmsProp(learning_rate=0.02)
    """
    from optax import rmsprop

    return rmsprop(learning_rate, decay=beta, eps=epscut, centered=centered)


_hide_submodules(__name__)
