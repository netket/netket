from ..abstract_optimizer import AbstractOptimizer

from jax.tree_util import tree_map


class Wrap(AbstractOptimizer):
    r"""Wrapper for Jax optimizers."""

    def __init__(self, machine, optimizer, repr_str=None):
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

        self._repr_str = repr_str

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
        repr_str = self._repr_str
        if repr_str is None:
            try:
                module = self._update_fun.__module__
                qname = self._update_fun.__qualname__
                repr_str = module + "." + qname

                if module.startswith("jax"):
                    repr_str = repr_str.rsplit(".", 2)[0]
            except:
                repr_str = ""

            rep = "Wrapper for the following Jax optimizer :\n  "
            rep += str(repr_str)
            return rep
