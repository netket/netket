from .abstract_optimizer import AbstractOptimizer


class Sgd(AbstractOptimizer):
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
    """

    def __init__(self, learning_rate, l2reg=0, decay_factor=1.0):
        r"""
            Constructs a new ``Sgd`` optimizer.

            Args:
               learning_rate (float): The learning rate :math:`\eta`
               l2_reg (float): The amount of :math:`L_2` regularization.
               decay_factor (float): The decay factor :math:`\gamma`.

            Examples:
               Simple SGD optimizer.

               >>> from netket.optimizer import Sgd
               >>> op = Sgd(learning_rate=0.05)
        """
        self._learning_rate = learning_rate
        self._l2reg = l2reg
        self._decay_factor = decay_factor
        self._eta = learning_rate

        if learning_rate <= 0:
            raise ValueError("Invalid learning rate.")
        if l2reg < 0:
            raise ValueError("Invalid L2 regularization.")
        if decay_factor < 1:
            raise ValueError("Invalid decay factor.")

    def update(self, grad, pars):
        self._eta *= self._decay_factor
        pars -= (grad + self._l2reg * pars) * self._eta
        return pars

    def reset(self):
        self._eta = self._learning_rate

    def __repr__(self):
        rep = "Sgd optimizer with these parameters :"
        rep += "\nLearning Rate = " + str(self._learning_rate)
        rep += "\nCurrent learning Rate = " + str(self._eta)
        rep += "\nL2 Regularization = " + str(self._l2reg)
        rep += "\nDecay Factor = " + str(self._decay_factor)
        return rep
