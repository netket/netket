# Copyright 2020, 2021 The NetKet Authors - All rights reserved.
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

import itertools as _itertools
import json

import numpy as _np
import scipy.integrate as _scint
from tqdm import tqdm

import netket.legacy as _nk
from netket.logging import JsonLog as _JsonLog
from jax.tree_util import tree_map as _tree_map

from netket.utils import (
    n_nodes as _n_nodes,
    node_number as _rank,
)

_NaN = float("NaN")


def _make_op(op, matrix_type):
    if matrix_type == "sparse":
        return op.to_sparse()
    elif matrix_type == "dense":
        return op.to_dense()
    elif matrix_type == "direct":
        return op.to_linear_op()


def _make_rhs(hamiltonian, propagation_type):
    if propagation_type == "real":

        def rhs(t, state):
            ham = hamiltonian(t)
            return -1j * ham.dot(state)

        return rhs

    elif propagation_type == "imaginary":

        def rhs(t, state):
            ham = hamiltonian(t)
            v0 = ham.dot(state)
            mean = _np.vdot(state, v0)
            return -v0 + mean * state

        return rhs


class _MockMachine:
    """
    A hack to make JsonLog work with full state data.
    """

    def __init__(self, state):
        self._state = state

    def save(self, filename):
        with open(filename, "w") as f:
            json.dump({"StateVector": [[z.real, z.imag] for z in self._state]}, f)


class PyExactTimePropagation:
    r"""
    Solver for exact real and imaginary time evolution, wrapping `scipy.integrate`
    for direct use within NetKet.
    """

    def __init__(
        self,
        hamiltonian,
        initial_state,
        dt,
        t0=0.0,
        propagation_type="real",
        matrix_type="sparse",
        solver=None,
        solver_kwargs={},
    ):
        r"""
        Create the exact time evolution driver.

           Args:
               :hamiltonian: Hamiltonian of the system.
               :initial_state: Initial state at t0.
               :dt (float): Simulation time step.
               :t0 (float): Initial time.
               :propagation_type: Specifies whether the imaginary or real-time
                    Schroedinger equation is solved. Should be one of "real" or
                    "imaginary".
               :matrix_type: The type of matrix used for the Hamiltonian when
                    creating the matrix wrapper. The default is `sparse`. The
                    other choices are `dense` and `direct`.
                :solver: A solver class, which should be a subclass of `scipy.integrate.OdeSolver`.
                    The default solver is `scipy.integrate.RK45`.
                :solver_kwargs (dict): Keyword arguments that are passed to the solver class,
                    e.g., for `atol` and `rtol`.

           Examples:
               Solving 1D Ising model with imaginary time propagation:

               ```python
               >>> import netket as nk
               >>> import numpy as np
               >>> L = 8
               >>> graph = nk.graph.Hypercube(L, n_dim=1, pbc=True)
               >>> hilbert = nk.hilbert.Spin(graph, 1/2)
               >>> n_states = hilbert.n_states
               >>> hamiltonian = nk.operator.Ising(hilbert, h=1.0)
               >>> psi0 = np.random.rand(n_states)
               >>> driver = nk.exact.PyExactTimePropagation(hamiltonian, stepper, t0=0,
               ...                                          initial_state=psi0,
               ...                                          propagation_type="imaginary")
               >>> for step in driver.iter(dt=0.05, n_iter=20):
               ...     print(driver.estimate(hamiltonian))
               ```
        """
        if isinstance(hamiltonian, _nk.operator.AbstractOperator):
            self._h_op = _make_op(hamiltonian, matrix_type)
            self._h = lambda t: self._h_op
        else:
            raise NotImplementedError("Time-dependent hHamiltonian not yet supported.")
        self._rhs = _make_rhs(self._h, propagation_type)

        self._mynode = _rank
        self._mpi_nodes = _n_nodes

        self.state = initial_state
        self._dt = dt
        self._t0 = t0
        self._t = t0
        self._steps = 0
        self._prop_type = propagation_type
        self._matrix_type = matrix_type

        Solver = solver if solver else _scint.RK45

        self._solver = Solver(
            self._rhs,
            self._t0,
            self.state,
            t_bound=_np.inf,
            max_step=dt,
            **solver_kwargs,
        )

    def _estimate_stats(self, obs):
        if isinstance(obs, _nk.operator.AbstractOperator):
            op = _make_op(obs, self._matrix_type)
        else:
            op = obs

        v0 = op.dot(self.state)
        mean = _np.vdot(self.state, v0)
        variance = _np.vdot(self.state, op.dot(v0)) - mean ** 2

        return _nk.stats.Stats(mean=mean, error_of_mean=0.0, variance=variance.real)

    def estimate(self, observables):
        return _tree_map(self._estimate_stats, observables)

    @property
    def t(self):
        r"""
        Current simulation time.
        """
        return self._t

    @property
    def dt(self):
        r"""
        Simulation time step.
        """
        return self._dt

    @property
    def step_count(self):
        return self._step

    def advance(self, n_steps):
        r"""
        Advance the time propagation by `n_steps` simulation steps
        of duration `self.dt`.

           Args:
               :n_steps (int): No. of steps to advance.
        """
        t_end = self._t + n_steps * self._dt

        self._solver.t = self.t
        self._solver.t_bound = t_end
        self._solver.status = "running"
        while self._solver.status == "running":
            self._solver.step()
        if self._solver.status == "failed":
            raise ...

        self.state = self._solver.y
        self._t = self._solver.t

    def iter(self, n_iter, step=1):
        """
        Returns a generator which advances the time evolution in
        steps of `step` for a total of `n_iter` times.

        Args:
            :n_iter (int): The total number of steps.
            :step (int=1): The size of each step.

        Yields:
            :(int): The current step.
        """
        for i in range(n_iter):
            # First yield, then step for compatibility with AbstractVariationalDriver
            yield i
            self.advance(step)

    def reset(self):
        """
        Resets the driver.
        """
        self.t0 = self.t
        self._step = 0

    def run(
        self,
        output_prefix,
        n_iter,
        obs=None,
        save_params_every=50,
        write_every=50,
        step_size=1,
        show_progress=True,
    ):
        """
        Executes thetime evolution for `n_iter` steps of `dt` and writing values of
        the observables `obs` to the output. The output are JSON file at
        `output_prefix.{log,state}`, overwriting files with the same prefix.

        Args:
            :output_prefix: The prefix at which JSON output should be stored.
            :n_iter: the total number of iterations
            :obs: An iterable containing all observables that should be computed
            :save_params_every: Every how many steps the parameters of the network should be
            serialized to disk (ignored if logger is provided)
            :write_every: Every how many steps the json data should be flushed to disk (ignored if
            logger is provided)
            :step_size: Every how many steps should observables be logged to disk (default=1)
            :show_progress: If true displays a progress bar (default=True)
        """
        if obs is None:
            obs = {}

        logger = _JsonLog(output_prefix, save_params_every, write_every)

        # Don't log on non-root nodes
        if self._mpi_nodes != 0:
            logger = None

        with tqdm(
            self.iter(n_iter, step_size), total=n_iter, disable=not show_progress
        ) as itr:
            for step in itr:
                # if the cost-function is defined then report it in the progress bar
                energy = self.estimate(self._h(self.t))
                itr.set_postfix_str(
                    "t={:.2e}, Energy={:.6e}".format(self.t, energy.mean.real)
                )

                obs_data = self.estimate(obs)
                obs_data["Energy"] = energy

                log_data = {}
                if self._loss_stats is not None:
                    obs_data[self._loss_name] = self._loss_stats
                log_data["Time"] = self.t

                if logger is not None:
                    logger(step, log_data, _MockMachine(self.state))

        # flush at the end of the evolution so that final values are saved to
        # file
        if logger is not None:
            logger.flush(_MockMachine(self.state))
