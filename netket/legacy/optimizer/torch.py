from .abstract_optimizer import AbstractOptimizer
import numpy as _np
import torch as _torch


class Torch(AbstractOptimizer):
    r"""Wrapper for Torch optimizers."""

    def __init__(self, machine, optimizer, **opt_pars):
        r"""
        Constructs a new ``Torch`` optimizer that can be used in NetKet drivers.

        Args:
            machine (AbstractMachine): The machine to be optimized.
            optimizer (torch.optim.Optimizer): A PyTorch optimizer.
            opt_pars: named parameters to be passed to opt at construction

        Examples:
            Simple SGD optimizer from PyTorch.

            >>> from netket.optimizer import Torch
            >>> from torch.optim import SGD
            >>>
            >>> g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ha = nk.operator.Ising(h=1.0, hilbert=hi)
            >>> ma = nk.machine.RbmReal(alpha=1, hilbert=hi)
            >>> learning_rate = 0.01
            >>> opt = Torch(ma, SGD, lr=learning_rate)
        """
        if not issubclass(optim, _torch.optim.Optimizer):
            raise ValueError("Not a valid Torch optimizer.")

        if machine.dtype is complex:
            raise ValueError(
                "Torch optimizers work only for machines with real parameters."
            )

        self._t_pars = _torch.from_numpy(machine.parameters.real)
        self._t_opt = optimizer([self._t_pars], **opt_pars)

        self.reset()

    def update(self, grad, pars):
        self._t_pars.grad = _torch.from_numpy(grad.real)
        self._t_opt.step()
        pars.real = self._t_pars.numpy()
        return pars

    def reset(self):
        self._t_opt.zero_grad()

    def __repr__(self):
        rep = "Wrapper for the following Torch optimizer :\n"
        rep += str(self._t_opt)
        return rep
