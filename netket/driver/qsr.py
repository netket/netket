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

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from numba import njit

from netket import jax as nkjax
from netket.operator import AbstractOperator, LocalOperator
from netket.stats import Stats
from netket.vqs import MCState
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils import mpi
from netket.utils.types import PyTree

from .vmc_common import info
from .abstract_variational_driver import AbstractVariationalDriver


def _build_rotation(hi, basis, dtype=complex):
    localop = LocalOperator(hi, constant=1.0, dtype=dtype)
    U_X = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, 1.0], [1.0, -1.0]])
    U_Y = 1.0 / (np.sqrt(2)) * np.asarray([[1.0, -1j], [1.0, 1j]])

    assert len(basis) == hi.size

    for (j, base) in enumerate(basis):
        if base == "X":
            localop *= LocalOperator(hi, U_X, [j])
        elif base == "Y":
            localop *= LocalOperator(hi, U_Y, [j])
        elif base == "Z" or base == "I":
            pass

    return localop


def convert_bases(Us):
    if not (isinstance(Us, list) or isinstance(Us, np.ndarray)):
        raise ValueError(
            "The bases should be a list or np.ndarray(dtype=object)" " of the bases."
        )

    if isinstance(Us[0], AbstractOperator):
        return Us

    if isinstance(Us[0], str):
        from netket.hilbert import Spin

        hilbert = Spin(0.5, N=len(Us[0]))
        N_samples = len(Us)

        _cache = {}
        _bases = np.empty(N_samples, dtype=object)

        for (i, basis) in enumerate(Us):
            if basis not in _cache:
                U = _build_rotation(hilbert, basis)
                _cache[basis] = U

            _bases[i] = _cache[basis]
        return _bases

    raise TypeError("Unknown type of measurement basis.")


def convert_data(σs, Us):
    """
    Converts samples and rotation operators to a more direct computational format.
    """
    Us = convert_bases(Us)

    N = σs.shape[-1]
    σs = σs.reshape(-1, N)
    Nb = σs.shape[0]

    # constant number of connected per opoerator
    Nc = Us[0].hilbert.local_size
    σp = np.zeros((0, N), dtype=σs.dtype)
    mels = np.zeros((0,), dtype=Us[0].dtype)
    secs = np.zeros(Nb + 1, dtype=np.intp)
    MAX_LEN = 0

    last_i = 0
    for (i, (σ, U)) in enumerate(zip(σs, Us)):
        σp_i, mels_i = U.get_conn(σ)
        # σp_i, mels_i = _remove_zeros(σp_i, mels_i)

        Nc = mels_i.size
        σp.resize((last_i + Nc, N))
        mels.resize((last_i + Nc,))
        σp[last_i:, :] = σp_i
        mels[last_i:] = mels_i
        secs[i] = last_i
        last_i = last_i + Nc
        MAX_LEN = max(Nc, MAX_LEN)

    # last
    σp.resize((last_i + MAX_LEN, N))
    mels.resize((last_i + MAX_LEN,))
    σp[last_i + Nc :, :] = 0.0
    mels[last_i + Nc :] = 0.0
    secs[-1] = last_i  # + MAX_LEN

    return σp, mels, secs, MAX_LEN


@njit
def _compose_sampled_data(
    σp, mels, secs, MAX_LEN, sampled_indices, min_padding_factor=128
):
    N_samples = sampled_indices.size
    N = σp.shape[-1]

    _σp = np.zeros((N_samples * MAX_LEN, N), dtype=σp.dtype)
    _mels = np.zeros((N_samples * MAX_LEN,), dtype=mels.dtype)
    _secs = np.zeros((N_samples + 1,), dtype=secs.dtype)
    _maxlen = 0

    last_i = 0
    for (n, i) in enumerate(sampled_indices):
        start_i, end_i = secs[i], secs[i + 1]
        len_i = end_i - start_i

        # print(f"{n}, {i}, {last_i}, {start_i}, {end_i}")
        _σp[last_i : last_i + len_i, :] = σp[start_i:end_i, :]
        _mels[last_i : last_i + len_i] = mels[start_i:end_i]

        last_i = last_i + len_i
        _secs[n + 1] = last_i

        _maxlen = max(_maxlen, len_i)

    padding_factor = max(
        MAX_LEN, min_padding_factor
    )  # minimum padding at 64 to avoid excessive recmpilation
    padded_size = padding_factor * int(np.ceil(last_i / padding_factor))

    _σp = _σp[:padded_size, :]
    _mels = _mels[:padded_size]

    return _σp, _mels, _secs, _maxlen


@partial(jax.jit, static_argnums=0)
def _avg_O(afun, pars, model_state, σ):
    σ = σ.reshape((-1, σ.shape[-1]))
    _, vjp = nkjax.vjp(lambda W: afun({"params": W, **model_state}, σ), pars)
    (O_avg,) = vjp(jnp.ones(σ.shape[0]) / σ.shape[0])
    return jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


####


@partial(jax.vmap, in_axes=(None, None, 0, 0, None), out_axes=0)
def map_masked_offset(fun, arr, offset, size, MAX_SIZE):
    masked_fun = jax.mask(fun, ("l",), "")
    return masked_fun((jax.lax.dynamic_slice(arr, [offset], [MAX_SIZE]),), dict(l=size))


def sum_sections(arr, secs, MAX_SIZE):
    """
    Equivalent to
    for i in range(secs.size-1):
        out[i] = jnp.sum(arr[secs[i]:secs[i+1]])

    secs must be 1 longer than the number of sections desired.
    MAX_SIZE is the max length of a section
    """
    offsets = secs[:-1]
    sizes = secs[1:] - offsets
    return map_masked_offset(jnp.sum, arr, offsets, sizes, MAX_SIZE)


@partial(jax.jit, static_argnums=(0, 5))
def local_value_rotated_kernel(logψ, pars, σp, mel, secs, MAX_LEN):
    logψ_σp = logψ(pars, σp)
    U_σσp_ψ_σp = mel * jnp.exp(logψ_σp)

    return jnp.log(sum_sections(U_σσp_ψ_σp, secs, MAX_LEN))


@partial(jax.jit, static_argnums=(0, 6))
def grad_local_value_rotated(logψ, pars, model_state, σp, mel, secs, MAX_LEN):
    log_val_rotated, vjp = nkjax.vjp(
        lambda W: local_value_rotated_kernel(
            logψ, {"params": W, **model_state}, σp, mel, secs, MAX_LEN
        ),
        pars,
    )
    log_val_rotated, _ = mpi.mpi_mean_jax(log_val_rotated)

    (O_avg,) = vjp(jnp.ones_like(log_val_rotated) / log_val_rotated.size)

    O_avg = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)

    return log_val_rotated, O_avg


@jax.jit
def compose_grads(grad_neg, grad_pos):
    return jax.tree_util.tree_multimap(
        lambda x, y: 2.0 * jnp.conj(x - y),
        grad_neg,
        grad_pos,
    )


# for nll
@partial(jax.jit, static_argnums=(0, 5))
def local_value_rotated_amplitude(logψ, pars, σp, mel, secs, MAX_LEN):
    logψ_σp = logψ(pars, σp)
    U_σσp_ψ_σp = mel * jnp.exp(logψ_σp)

    return jnp.log(jnp.abs(sum_sections(U_σσp_ψ_σp, secs, MAX_LEN)) ** 2)


#####


@jax.jit
def tree_norm(a):
    return nkjax.tree_dot(jax.tree_map(jnp.conj, a), a)


####


class QSR(AbstractVariationalDriver):
    """
    Energy minimization using Variational Monte Carlo (VMC).
    """

    # TODO docstring
    def __init__(
        self,
        training_data,
        training_batch_size,
        optimizer,
        *args,
        variational_state=None,
        preconditioner: PreconditionerT = identity_preconditioner,
        seed=None,
        **kwargs,
    ):
        """
        Initializes the driver class.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the
                bare energy gradient.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        if variational_state is None:
            variational_state = MCState(*args, **kwargs)

        super().__init__(variational_state, optimizer, minimized_quantity_name="KL")

        self.preconditioner = preconditioner

        self._dp = None  # type: PyTree
        self._S = None
        self._sr_info = None

        if not isinstance(training_data, tuple) or len(training_data) != 2:
            raise TypeError("not a tuple of length 2")

        self._rng = np.random.default_rng(
            np.asarray(nkjax.mpi_split(nkjax.PRNGKey(seed)))
        )

        σp, mels, secs, MAX_LEN = convert_data(*training_data)
        self._training_samples, self._training_rotations = training_data
        self._training_σp = σp
        self._training_mels = mels
        self._training_secs = secs
        self._training_max_len = MAX_LEN
        self._training_samples_n = len(secs) - 1

        self.training_batch_size = training_batch_size

    def _forward_and_backward(self):
        """
        Performs a number of VMC optimization steps.

        Args:
            n_steps (int): Number of steps to perform.
        """
        state = self.state

        state.reset()

        self._grad_neg = _avg_O(
            state._apply_fun,
            state.parameters,
            state.model_state,
            state.samples,
        )

        # sample training data for pos grad
        self._sampled_indices = np.sort(
            self._rng.integers(
                self._training_samples_n, size=(self.training_batch_size,)
            )
        )

        # compose data
        self._σp, self._mels, self._secs, self._maxlen = _compose_sampled_data(
            self._training_σp,
            self._training_mels,
            self._training_secs,
            self._training_max_len,
            self._sampled_indices,
        )

        _log_val_rot, self._grad_pos = grad_local_value_rotated(
            state._apply_fun,
            state.parameters,
            state.model_state,
            self._σp,
            self._mels,
            self._secs,
            self._maxlen,
        )

        self._loss_grad = compose_grads(self._grad_neg, self._grad_pos)

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_multimap(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    def nll(self):
        log_val_rot = local_value_rotated_amplitude(
            self.state._apply_fun,
            self.state.variables,
            self._σp,
            self._mels,
            self._secs,
            self._maxlen,
        )

        ce = jnp.mean(log_val_rot)
        # mpi

        # log norm calculation
        log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        return log_n - ce

    def _log_additional_data(self, log_dict, step):
        log_dict["loss_grad_norm"] = tree_norm(self._loss_grad)
        log_dict["dp_norm"] = tree_norm(self._dp)

    @property
    def energy(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    def __repr__(self):
        return (
            "QSR("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )

    def info(self, depth=0):
        lines = [
            "{}: {}".format(name, info(obj, depth=depth + 1))
            for name, obj in [
                ("Hamiltonian ", self._ham),
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
                ("State       ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
