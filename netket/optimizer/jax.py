from .abstract_optimizer import AbstractOptimizer
import jax


class Jax(AbstractOptimizer):
    r"""Wrapper for Jax optimizers.
    """

    def __init__(self, machine, optimizer):
        r"""
           Constructs a new ``Jax`` optimizer that can be used in NetKet drivers.

           Args:
               machine (AbstractMachine): The machine to be optimized.
               optimizer (jax.experimental.optimizers): A Jax optimizer.

           Examples:
               Simple SGD optimizer from Jax.

               >>> from netket.optimizer import Jax
               >>> from jax.experimental.optimizers import sgd as Sgd
               >>>

        """
        self._init_fun, self._update_fun, self._params_fun = optimizer

        self._machine = machine

        self.reset()

    def update(self, grad, pars):
        self._opt_state = self._update_fun(self._step, grad, self._opt_state)
        self._step += 1
        pars = self._params_fun(self._opt_state)

        return pars

    def reset(self):
        self._step = 0
        self._opt_state = self._init_fun(self._machine.parameters)

    def __repr__(self):
        rep = "Wrapper for the following Jax optimizer :\n"
        rep += str(self._j_opt)
        return rep
