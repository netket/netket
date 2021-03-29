from ..abstract_optimizer import AbstractOptimizer


class Sgd(AbstractOptimizer):
    def __init__(self, learning_rate, l2reg=0, decay_factor=1.0):

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
        if self._learning_rate != self._eta:
            lr = "learning_rate[base]={}, learning_rate[current]={}".format(
                self._learning_rate, self._eta
            )
        else:
            lr = "learning_rate={}".format(self._learning_rate)
        return ("Sgd({}, l2reg={}, decay_factor={})").format(
            lr, self._l2reg, self._decay_factor
        )
