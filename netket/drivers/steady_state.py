import math

import jax
from jax.tree_util import tree_map
import jax.numpy as jnp

from netket.operator import local_values as _local_values, Squared
from netket.stats import (
    statistics as _statistics,
    mean as _mean,
    sum_inplace as _sum_inplace,
)

from netket.variational import MCMixedState

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver


class SteadyState(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    def __init__(
        self, lindbladian, optimizer, *args, variational_state=None, sr=None, **kwargs
    ):
        """
        Initializes the driver class.

        Args:
            lindbladian (AbstractOperator): The Hamiltonian of the system.
            optimizer (AbstractOptimizer): Determines how optimization steps are performed given the
                bare energy gradient.
            sr (SR, optional): Determines whether and how stochastic reconfiguration
                is applied to the bare energy gradient before performing applying
                the optimizer. If this parameter is not passed or None, SR is not used.

        Example:
            Optimizing a 1D wavefunction with Variational Monte Carlo.

            >>> import netket as nk
            >>> SEED = 3141592
            >>> g = nk.graph.Hypercube(length=8, n_dim=1)
            >>> hi = nk.hilbert.Spin(s=0.5, graph=g)
            >>> ma = nk.machine.RbmSpin(hilbert=hi, alpha=1)
            >>> ma.init_random_parameters(seed=SEED, sigma=0.01)
            >>> ha = nk.operator.Ising(hi, h=1.0)
            >>> sa = nk.sampler.MetropolisLocal(machine=ma)
            >>> op = nk.optimizer.Sgd(learning_rate=0.1)
            >>> vmc = nk.Vmc(ha, sa, op, 200)

        """
        if variational_state is None:
            variational_state = MCMixedState(*args, **kwargs)

        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

        self._lind = lindbladian
        self._ldag_l = Squared(lindbladian)

        self.sr = sr

        self._dp = None

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """

        self.state.reset()

        # Compute the local energy estimator and average Energy
        self._loss_stats, self._loss_grad = self.state.expect_and_grad(self._ldag_l)

        if self.sr is not None:
            self._S = self.state.quantum_geometric_tensor(self.sr)
            self._dp = self._S(self._loss_grad)
        else:
            # tree_map(lambda x, y: x if is_ccomplex(y) else x.real, self._grads, self.state.parameters)
            self._dp = self._loss_grad

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_multimap(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    @property
    def ldagl(self):
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    #    def reset(self):
    #        super().reset()

    def __repr__(self):
        return "SteadyState(step_count={}, n_samples={}, n_discard={})".format(
            self.step_count, self.n_samples, self.n_discard
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Lindbladian ", self._lind),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
