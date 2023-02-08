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

from typing import List, Optional, Tuple, Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from numba import njit

from netket import jax as nkjax
from netket.driver import AbstractVariationalDriver
from netket.driver.vmc_common import info
from netket.operator import AbstractOperator, LocalOperator
from netket.hilbert import AbstractHilbert, Spin
from netket.vqs import VariationalState
from netket.vqs import ExactState
from netket import nn
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)
from netket.utils import mpi
from netket.utils.types import PyTree, Scalar, DType, Array
from netket.utils.dispatch import dispatch

from netket.stats import statistics

BaseType = Union[AbstractOperator, np.ndarray, str]


def _build_rotation(
    hi: Spin, basis: Union[List, str], dtype: Optional[DType] = np.complex64
) -> AbstractOperator:
    r"""
    Construct basis rotation operators from a Pauli string of "X", "Y", "Z" and "I".

    Args:
        hi: The Hilbert space
        basis: The Pauli string
        dtype: The data type of the returned operator

    Returns:
        The rotation operator
    """
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


def _check_bases_type(Us: Union[List[BaseType], np.ndarray]) -> List[AbstractOperator]:
    r"""
    Check if the given bases are valid for the quantum state reconstruction driver.

    Args:
        Us (list or np.ndarray): A list of bases

    Raises:
        ValueError: When not given a list or np.ndarray
        TypeError: If the type of the operators is not a child of AbstractOperator
    """
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


def _convert_data(
    sigma_s: np.ndarray,
    Us: Union[List[BaseType], np.ndarray],
    mixed_states: Optional[bool] = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Convert sampled states and rotation operators to a more direct computational format.
    Specifically, for each sampled state sigma_s, find all the states sigma_p that have non-zero
    matrix elements with sigma_s in the rotation operators Us. The corresponding
    non-zero matrix elements (mels) and the indices of sigma_p and Us that divide
    different sigma_s (secs) are also returned.

    For pure states, we directly return sigma_p, mels and secs, so that one can use them
    to compute <sigma_s|U|sigma_p> and sum over sigma_p. For mixed states, we return sigma_p x sigma_p,
    mels and mels.conj() and the corresponding secs, so that one can use them to compute
    <sigma_s|U|sigma_p><sigma_p'|U|sigma_s> and sum over both sigma_p and sigma_p'.

    Args:
        sigma_s (np.ndarray): The states
        Us (np.ndarray or list): The list of rotations
        mixed_states (bool): Whether to use mixed states or not

    Returns:
        sigma_p (np.ndarray): All the states that have non-zero matrix elements
        with the input states sigma_s in the rotation operators Us.
        mels (np.ndarray): The corresponding non-zero matrix elements.
        secs (np.ndarray): Indices of sigma_p and Us that divide different sigma_s.
        MAX_LEN (int): The maximum number of connected states.
    """
    Us = _check_bases_type(Us)

    # Error message when user tries to convert less or more sigmas than Us
    assert (
        sigma_s.shape[0] == Us.shape[0]
    ), "The number of samples should be equal to the number of rotations."

    N = sigma_s.shape[-1]
    sigma_s = sigma_s.reshape(-1, N)
    Nb = sigma_s.shape[0]

    # constant number of connected states per operator
    Nc = Us[0].hilbert.local_size
    if mixed_states:
        sigma_p = np.zeros((0, 2 * N), dtype=sigma_s.dtype)
    else:
        sigma_p = np.zeros((0, N), dtype=sigma_s.dtype)
    mels = np.zeros((0,), dtype=Us[0].dtype)
    secs = np.zeros(Nb + 1, dtype=np.intp)
    MAX_LEN = 0

    last_i = 0
    for (i, (sigma, U)) in enumerate(zip(sigma_s, Us)):
        sigma_p_i, mels_i = U.get_conn(sigma)
        if mixed_states:
            # size of the cartesian product sigma_p x sigma_p
            Nc = mels_i.size**2
            sigma_p = np.resize(sigma_p, (last_i + Nc, 2 * N))
            mels = np.resize(mels, (last_i + Nc,))
            # indices of the cartesian product
            x, y = np.meshgrid(np.arange(mels_i.size), np.arange(mels_i.size))
            sigma_p[last_i:, :] = np.hstack(
                [sigma_p_i[x.flatten()], sigma_p_i[y.flatten()]]
            )
            # <sigma_s|U|sigma_p><sigma_p'|U|sigma_s>
            mels[last_i:] = np.prod(
                np.stack(
                    [mels_i[x.flatten()], np.conjugate(mels_i[y.flatten()])], axis=-1
                ),
                axis=-1,
            )
        else:
            Nc = mels_i.size
            sigma_p = np.resize(sigma_p, (last_i + Nc, N))
            mels = np.resize(mels, (last_i + Nc,))
            sigma_p[last_i:, :] = sigma_p_i
            # <sigma_s|U|sigma_p>
            mels[last_i:] = mels_i
        secs[i] = last_i
        last_i = last_i + Nc
        MAX_LEN = max(Nc, MAX_LEN)

    # last
    if mixed_states:
        sigma_p = np.resize(sigma_p, (last_i + MAX_LEN, 2 * N))
    else:
        sigma_p = np.resize(sigma_p, (last_i + MAX_LEN, N))
    mels = np.resize(mels, (last_i + MAX_LEN,))
    sigma_p[last_i + Nc :, :] = 0.0
    mels[last_i + Nc :] = 0.0
    secs[-1] = last_i  # + MAX_LEN

    return sigma_p, mels, secs, MAX_LEN


@njit
def _compose_sampled_data(
    sigma_p: np.ndarray,
    mels: np.ndarray,
    secs: np.ndarray,
    MAX_LEN: int,
    sampled_indices: np.ndarray,
    min_padding_factor: Optional[int] = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""
    Given the sampled indices, select the corresponding data from sigma_p, mels and secs.

    Args:
        sigma_p (np.ndarray): All the states that have non-zero matrix elements
        with the input states sigma_s in the rotation operators Us.
        mels (np.ndarray): The corresponding non-zero matrix elements.
        secs (np.ndarray): Indices of sigma_p and Us that divide different sigma_s.
        MAX_LEN (int): The maximum number of connected states.
        sampled_indices (np.ndarray): The indices of the sampled states.
        min_padding_factor (int): The minimum padding factor.

    Returns:
        The sampled sigma_p, mels, secs and MAX_LEN.
    """
    N_samples = sampled_indices.size
    N = sigma_p.shape[-1]

    _sigma_p = np.zeros((N_samples * MAX_LEN, N), dtype=sigma_p.dtype)
    _mels = np.zeros((N_samples * MAX_LEN,), dtype=mels.dtype)
    _secs = np.zeros((N_samples + 1,), dtype=secs.dtype)
    _maxlen = 0

    last_i = 0
    for (n, i) in enumerate(sampled_indices):
        start_i, end_i = secs[i], secs[i + 1]
        len_i = end_i - start_i

        # print(f"{n}, {i}, {last_i}, {start_i}, {end_i}")
        _sigma_p[last_i : last_i + len_i, :] = sigma_p[start_i:end_i, :]
        _mels[last_i : last_i + len_i] = mels[start_i:end_i]

        last_i = last_i + len_i
        _secs[n + 1] = last_i

        _maxlen = max(_maxlen, len_i)

    padding_factor = max(
        MAX_LEN, min_padding_factor
    )  # minimum padding at 64 to avoid excessive recompilation
    padded_size = padding_factor * int(np.ceil(last_i / padding_factor))

    _sigma_p = _sigma_p[:padded_size, :]
    _mels = _mels[:padded_size]

    return _sigma_p, _mels, _secs, _maxlen


@partial(jax.jit, static_argnums=(0))
def _avg_O(afun, pars, model_state, sigma):
    r"""
    Compute the gradient of the normalization log Z by sample mean.
    For example, for an unnormalized probability distribution p_i, we have
    :math:`\nabla log Z = \frac{1}{Z} \sum_i \nabla p_i = \sum_i \frac{1}{Z} p_i \nabla \log p_i \approx \frac{1}{N} \sum_i \nabla \log p_i`

    Args:
        afun (function): The unnormalized log probability function.
        pars (PyTree): The parameters of the model.
        model_state (PyTree): The model state.
        sigma (np.ndarray): The sampled states.

    Returns:
        The gradient of the normalization log Z.
    """
    sigma = sigma.reshape((-1, sigma.shape[-1]))
    _, vjp = nkjax.vjp(lambda W: afun({"params": W, **model_state}, sigma), pars)
    (O_avg,) = vjp(jnp.ones(sigma.shape[0]) / sigma.shape[0])
    return jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


@dispatch
def _grad_negative(state_diag):
    r"""
    Compute the gradient of the normalization log Z by sample mean.
    For example, for an unnormalized probability distribution p_i, we have
    :math:`\nabla log Z = \frac{1}{Z} \sum_i \nabla p_i = \sum_i \frac{1}{Z} p_i \nabla \log p_i \approx \frac{1}{N} \sum_i \nabla \log p_i`

    Args:
        state_diag (VariationalState): The variational state.

    Returns:
        The gradient of the normalization log Z.
    """
    return _avg_O(
        state_diag._apply_fun,
        state_diag.parameters,
        state_diag.model_state,
        state_diag.samples,
    )


@partial(jax.jit, static_argnums=(0, 1))
def _avg_O_exact(hilbert: AbstractHilbert, afun, pars, model_state):
    r"""
    Same as _avg_O, but for ExactState.
    """
    sigma = hilbert.all_states()
    sigma = sigma.reshape((-1, sigma.shape[-1]))
    _, vjp = nkjax.vjp(lambda W: afun({"params": W, **model_state}, sigma), pars)
    psi_2 = jnp.abs(jnp.exp(afun({"params": pars, **model_state}, sigma))) ** 2
    psi_2 /= jnp.sum(psi_2)
    (O_avg,) = vjp(psi_2)
    return jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


@dispatch
def _grad_negative(state_diag: ExactState):
    r"""
    Same as _grad_negative, but for ExactState.
    """
    return _avg_O_exact(
        state_diag._hilbert,
        state_diag._apply_fun,
        state_diag.parameters,
        state_diag.model_state,
    )


####


def _sum_sections(arr: Array, secs: Array) -> Array:
    """
    Sum the elements of arr in sections defined by secs.
    Equivalent to
    for i in range(secs.size-1):
        out[i] = jnp.sum(arr[secs[i]:secs[i+1]])

    secs must be 1 longer than the number of sections desired.
    MAX_SIZE is the max length of a section

    Args:
        arr (np.ndarray): The array to be summed.
        secs (np.ndarray): The indices that define the sections.

    Returns:
        The sum of the array in sections.
    """
    offsets = secs[:-1]
    sizes = secs[1:] - offsets

    N_rows = secs.size - 1
    # Construct the indices necessary to perform the segment_sum
    indices = jnp.repeat(jnp.arange(N_rows), sizes, total_repeat_length=arr.size)

    return jax.ops.segment_sum(
        arr, indices, num_segments=N_rows, indices_are_sorted=True
    )


@partial(jax.jit, static_argnums=(0))
def _local_value_rotated_kernel(log_psi, pars, sigma_p, mel, secs):
    r"""
    Compute the log probability amplitude \log <sigma_p|U|psi> of obtaining an outcome state sigma_p in the rotated basis.
    For mixed states, it's \log <sigma_p|U \rho U^\dagger|sigma_p>.

    Args:
        log_psi (function): The log wavefunction or density matrix.
        pars (PyTree): The parameters of the model.
        sigma_p (np.ndarray): The sampled states.
        mel (np.ndarray): The matrix elements of the rotations.
        secs (np.ndarray): The indices that define the connected sections of different basis.

    Returns:
        The probability amplitude of obtaining an outcome state sigma_p in the rotated basis.
    """
    log_psi_sigma_p = log_psi(pars, sigma_p)
    U_sigma_sigma_p_psi_sigma_p = mel * jnp.exp(log_psi_sigma_p)

    return jnp.log(_sum_sections(U_sigma_sigma_p_psi_sigma_p, secs))


@partial(jax.jit, static_argnums=(0))
def _grad_local_value_rotated(log_psi, pars, model_state, sigma_p, mel, secs):
    r"""
    Compute the gradient of the log probability amplitude \log <sigma_p|U|psi> of obtaining an outcome state sigma_p in the rotated basis.
    For mixed states, it's the gradient of \log <sigma_p|U \rho U^\dagger|sigma_p>.

    Args:
        log_psi (function): The log wavefunction or density matrix.
        pars (PyTree): The parameters of the model.
        model_state (PyTree): The model state.
        sigma_p (np.ndarray): The sampled states.
        mel (np.ndarray): The matrix elements of the rotations.
        secs (np.ndarray): The indices that define the connected sections of different basis.

    Returns:
        The gradient of the probability amplitude of obtaining an outcome state sigma_p in the rotated basis.
    """
    log_val_rotated, vjp = nkjax.vjp(
        lambda W: _local_value_rotated_kernel(
            log_psi, {"params": W, **model_state}, sigma_p, mel, secs
        ),
        pars,
    )
    log_val_rotated, _ = mpi.mpi_mean_jax(log_val_rotated)

    (O_avg,) = vjp(jnp.ones_like(log_val_rotated) / log_val_rotated.size)

    O_avg = jax.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)

    return log_val_rotated, O_avg


@jax.jit
def _compose_grads(grad_neg, grad_pos):
    r"""
    Compose the negative gradient (the normalization log Z) and the positive gradient (log probability amplitude).
    For complex parameters, the complex conjugate is taken.

    Args:
        grad_neg (PyTree): The negative gradient.
        grad_pos (PyTree): The positive gradient.

    Returns:
        The composed gradient.
    """
    return jax.tree_util.tree_map(
        lambda x, y: jnp.conj(x - y),
        grad_neg,
        grad_pos,
    )


# for nll
@partial(jax.jit, static_argnums=(0,))
def _local_value_rotated_amplitude(log_psi, pars, sigma_p, mel, secs):
    r"""
    Only for monitoring negative log likelihood.

    Compute the log probability amplitude squared \log |<sigma_p|U|psi>|^2 of obtaining an outcome state sigma_p in the rotated basis.
    For mixed states, it's <sigma_p|U \rho U^\dagger|sigma_p>.

    Args:
        log_psi (function): The log wavefunction or density matrix.
        pars (PyTree): The parameters of the model.
        sigma_p (np.ndarray): The sampled states.
        mel (np.ndarray): The matrix elements of the rotations.
        secs (np.ndarray): The indices that define the connected sections of different basis.

    Returns:
        The probability amplitude of obtaining an outcome state sigma_p in the rotated basis.
    """
    log_psi_sigma_p = log_psi(pars, sigma_p)
    U_sigma_sigma_p_psi_sigma_p = mel * jnp.exp(log_psi_sigma_p)

    return jnp.log(jnp.abs(_sum_sections(U_sigma_sigma_p_psi_sigma_p, secs)) ** 2)


####


class QSR(AbstractVariationalDriver):
    """
    The quantum state reconstruction driver minimizing KL divergence.
    """

    def __init__(
        self,
        training_data: Tuple[List, List],
        training_batch_size: int,
        optimizer,
        *,
        variational_state: VariationalState,
        preconditioner: Optional[PreconditionerT] = identity_preconditioner,
        seed: Optional[int] = None,
        batch_sample_no_replace: Optional[bool] = False,
        control_variate_update_freq: Optional[
            Union[
                int,
                str,
            ]
        ] = None,
        chunk_size: Optional[int] = None,
    ):
        """Initializes the QSR driver class.

        Args:
            training_data: A tuple of two arrays (sigma_s, Us). sigma_s is a the
                sampled states and Us is the corresponding rotations.
            training_batch_size: The training batch size.
            optimizer: The optimizer to use. You can use optax optimizers or
                choose from the predefined optimizers netket offers.
            variational_state: The variational state to optimize.
            preconditioner: The preconditioner to use.
                Defaults to identity_preconditioner.
            seed: The RNG seed. Defaults to None.
            batch_sample_no_replace: Whether to sample without replacement. Defaults to False.
            control_variate_update_freq: The frequency of updating the control variates. Defaults to None.
                "Adaptive" for adaptive update frequency, i.e. n_samples // batch size.
            chunk_size: The chunk size for the control variates. Defaults to None.

        Raises:
            Warning: If the chunk size is not divisor of the training data size.
            TypeError: If the training data is not a 2 element tuple.
        """

        super().__init__(variational_state, optimizer)

        self.preconditioner = preconditioner

        if not isinstance(training_data, tuple) or len(training_data) != 2:
            raise TypeError("not a tuple of length 2")

        self._rng = np.random.default_rng(
            np.asarray(nkjax.mpi_split(nkjax.PRNGKey(seed)))
        )

        # mixed states
        self.mixed_states = variational_state.__class__.__name__ in ["MCMixedState"]

        self.batch_sample_no_replace = batch_sample_no_replace

        sigma_p, mels, secs, MAX_LEN = _convert_data(*training_data, self.mixed_states)
        self._training_rotations = training_data[1]
        self._training_sigma_p = sigma_p
        self._training_mels = mels
        self._training_secs = secs
        self._training_max_len = MAX_LEN
        self._training_samples_n = len(secs) - 1

        self.training_batch_size = training_batch_size

        # statistical constants
        self._entropy = None

        # control variates
        if control_variate_update_freq == "Adaptive":
            if self._training_samples_n <= training_batch_size:
                self._control_variate_update_freq = None
            else:
                self._control_variate_update_freq = (
                    self._training_samples_n // training_batch_size
                )
        else:
            self._control_variate_update_freq = control_variate_update_freq
        self._control_variate_expectation = None
        self._control_variate_params = None
        self._chunk_size = chunk_size

        # chunk
        if self._chunk_size is not None:
            self.n_chunk = self._training_samples_n // self._chunk_size
            if not self.n_chunk * self._chunk_size == self._training_samples_n:
                print(
                    "WARNING: chunk size does not divide the number of samples, the last few chunks will be smaller"
                )
            self._chunked_indices = np.array_split(
                np.arange(self._training_samples_n), self.n_chunk
            )

    def _forward_and_backward(self):
        state = self.state

        if self.mixed_states:
            state_diag = state.diagonal
        else:
            state_diag = self.state

        state.reset()

        # compute the neg gradient of log Z
        self._grad_neg = _grad_negative(state_diag)

        # sample training data for pos grad
        self._sampled_indices = np.sort(
            self._rng.choice(
                self._training_samples_n,
                size=(self.training_batch_size,),
                replace=not self.batch_sample_no_replace,
            )
        )

        # compose data
        self._sigma_p, self._mels, self._secs, self._maxlen = _compose_sampled_data(
            self._training_sigma_p,
            self._training_mels,
            self._training_secs,
            self._training_max_len,
            self._sampled_indices,
        )

        # compute the pos gradient of log p
        _log_val_rot, self._grad_pos = _grad_local_value_rotated(
            state._apply_fun,
            state.parameters,
            state.model_state,
            self._sigma_p,
            self._mels,
            self._secs,
        )

        # control variates
        if self._control_variate_update_freq is not None:
            # update control variate
            if self.step_count % self._control_variate_update_freq == 0:
                if self._chunk_size is not None:
                    for i in range(self.n_chunk):
                        (
                            sigma_p_chunk,
                            mels_chunk,
                            secs_chunk,
                            _,
                        ) = _compose_sampled_data(
                            self._training_sigma_p,
                            self._training_mels,
                            self._training_secs,
                            self._training_max_len,
                            self._chunked_indices[i],
                        )
                        if i == 0:
                            # chunking: initialize variable
                            self._control_variate_expectation = jax.tree_util.tree_map(
                                lambda y: y * len(self._chunked_indices[i]),
                                _grad_local_value_rotated(
                                    state._apply_fun,
                                    state.parameters,
                                    state.model_state,
                                    sigma_p_chunk,
                                    mels_chunk,
                                    secs_chunk,
                                )[1],
                            )
                        else:
                            # chunking: accumulate
                            self._control_variate_expectation = jax.tree_util.tree_map(
                                lambda x, y: x + y * len(self._chunked_indices[i]),
                                self._control_variate_expectation,
                                _grad_local_value_rotated(
                                    state._apply_fun,
                                    state.parameters,
                                    state.model_state,
                                    sigma_p_chunk,
                                    mels_chunk,
                                    secs_chunk,
                                )[1],
                            )
                    # chunking: average
                    self._control_variate_expectation = jax.tree_util.tree_map(
                        lambda x: x / self._training_samples_n,
                        self._control_variate_expectation,
                    )
                else:
                    self._control_variate_expectation = _grad_local_value_rotated(
                        state._apply_fun,
                        state.parameters,
                        state.model_state,
                        self._training_sigma_p,
                        self._training_mels,
                        self._training_secs,
                    )[1]
                self._control_variate_params = state.parameters

            # control variate gradient
            # it's the graident evaluated at an earlier point
            _, self._grad_pos_cv = _grad_local_value_rotated(
                state._apply_fun,
                self._control_variate_params,
                state.model_state,
                self._sigma_p,
                self._mels,
                self._secs,
            )

            # gather gradient
            # grad <- grad - grad_cv + E[grad_cv]
            self._grad_pos = jax.tree_map(
                lambda x, y, Ey: x - y + Ey,
                self._grad_pos,
                self._grad_pos_cv,
                self._control_variate_expectation,
            )

        # compose neg and pos gradient
        # and take complex conjugate
        self._loss_grad = _compose_grads(self._grad_neg, self._grad_pos)

        # restore the square in prob = |psi|^2
        if not self.mixed_states:
            self._loss_grad = jax.tree_map(lambda x: x * 2.0, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._loss_grad = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._loss_grad,
            self.state.parameters,
        )

        # if it's the identity it does
        # self._dp = self._loss_grad
        self._dp = self.preconditioner(self.state, self._loss_grad)

        # If parameters are real, then take only real part of the gradient (if it's complex)
        self._dp = jax.tree_map(
            lambda x, target: (x if jnp.iscomplexobj(target) else x.real),
            self._dp,
            self.state.parameters,
        )

        return self._dp

    def nll(self, return_stats: Optional[bool] = True):
        r"""
        Compute the Negative-Log-Likelihood over a batch of data.

        Args:
            return_stats: if True, return the statistics.

        .. warn::

            Exponentially expensive in the hilbert space size!

        """
        log_val_rot = _local_value_rotated_amplitude(
            self.state._apply_fun,
            self.state.variables,
            self._sigma_p,
            self._mels,
            self._secs,
        )
        if self.mixed_states:
            log_val_rot /= 2

        ce = log_val_rot

        # log norm calculation
        if self.mixed_states:
            log_psi = (
                self.state.diagonal.log_value(self.state.hilbert_physical.all_states())
                / 2
            )
        else:
            log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        if return_stats:
            return statistics(jnp.real(log_n - ce))
        return jnp.real(log_n - ce)

    def nll_whole_training_set(self, return_stats: Optional[bool] = True):
        r"""
        Compute the Negative-Log-Likelihood over the whole training set.

        Args:
            return_stats: if True, return the statistics.

        .. warn::

            Exponentially expensive in the hilbert space size!
        """
        if self._chunk_size is not None:
            for i in range(self.n_chunk):
                sigma_p_chunk, mels_chunk, secs_chunk, _ = _compose_sampled_data(
                    self._training_sigma_p,
                    self._training_mels,
                    self._training_secs,
                    self._training_max_len,
                    self._chunked_indices[i],
                )
                if i == 0:
                    # chunking: initialize variable
                    log_val_rot = _local_value_rotated_amplitude(
                        self.state._apply_fun,
                        self.state.variables,
                        sigma_p_chunk,
                        mels_chunk,
                        secs_chunk,
                    )
                else:
                    # chunking: accumulate
                    log_val_rot = jnp.concatenate(
                        [
                            log_val_rot,
                            _local_value_rotated_amplitude(
                                self.state._apply_fun,
                                self.state.variables,
                                sigma_p_chunk,
                                mels_chunk,
                                secs_chunk,
                            ),
                        ]
                    )
        else:
            log_val_rot = _local_value_rotated_amplitude(
                self.state._apply_fun,
                self.state.variables,
                self._training_sigma_p,
                self._training_mels,
                self._training_secs,
            )

        # square root <sigma|rho|sigma> to keep in line with the pure state case
        if self.mixed_states:
            log_val_rot /= 2

        ce = log_val_rot

        # log norm calculation
        if self.mixed_states:
            log_psi = (
                self.state.diagonal.log_value(self.state.hilbert_physical.all_states())
                / 2
            )
        else:
            log_psi = self.state.log_value(self.state.hilbert.all_states())
        maxl = log_psi.real.max()
        log_n = jnp.log(jnp.exp(2 * (log_psi.real - maxl)).sum()) + 2 * maxl

        # result
        if return_stats:
            return statistics(jnp.real(log_n - ce))
        return jnp.real(log_n - ce)

    def entropy(
        self,
        target_state: Array,
        n_shots: Optional[int] = 1,
        no_cache: Optional[bool] = False,
    ) -> float:
        r"""
        Compute the average entropy of the probability distributions
        given by the target state in different measurement basis.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.
            no_cache: if True, do not use the cached value.

        .. warn::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        if len(target_state.shape) == 1:
            target_state = target_state.reshape(-1, 1)
            target_state = target_state @ target_state.conj().T
        if self._entropy is not None and not no_cache:
            return self._entropy
        rotations = self._training_rotations[::n_shots]
        entropy_list = []
        for rot in rotations:
            rho_rot = (rot @ (rot @ target_state).conj().T).conj().T
            prob = np.real(np.diag(rho_rot))
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            entropy_list.append(entropy)
        if no_cache:
            return np.mean(entropy_list)
        self._entropy = np.mean(entropy_list)
        return self._entropy

    def KL(self, target_state: Optional[Array] = None, n_shots: Optional[int] = None):
        r"""
        Compute average KL divergence loss over a batch of data.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warn::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        return statistics(
            self.nll(return_stats=False) - self.entropy(target_state, n_shots)
        )

    def KL_whole_training_set(
        self, target_state: Optional[Array] = None, n_shots: Optional[int] = None
    ):
        r"""
        Compute average KL divergence loss over the whole training set.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warn::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        return statistics(
            self.nll_whole_training_set(return_stats=False)
            - self.entropy(target_state, n_shots)
        )

    def KL_exact(
        self, target_state: Optional[Array] = None, n_shots: Optional[int] = 1
    ) -> float:
        r"""
        Compute the average KL divergence loss between the variational state and the target state.

        Args:
            target_state: the target state.
            n_shots: number of shots per measurement basis.

        .. warn::

            Exponentially expensive in the hilbert space size!

            Need to know the target state!

        """
        if len(target_state.shape) == 1:
            target_state = target_state.reshape(-1, 1)
            target_state = target_state @ target_state.conj().T
        rotations = self._training_rotations[::n_shots]
        KL_list = []
        try:
            vs = self.state.to_matrix(normalize=True)
        except:
            vs = self.state.to_array(normalize=True).reshape(-1, 1)
            vs = vs @ vs.conj().T
        for rot in rotations:
            rho_rot = (rot @ (rot @ vs).conj().T).conj().T
            target_rot = (rot @ (rot @ target_state).conj().T).conj().T
            prob = np.maximum(np.real(np.diag(rho_rot)), 0)
            prob_target = np.maximum(np.real(np.diag(target_rot)), 0)
            KL = np.sum(prob_target * np.log(prob_target / (prob + 1e-10) + 1e-10))
            KL_list.append(KL)
        return np.mean(KL_list)

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
                ("Optimizer   ", self._optimizer),
                ("SR solver   ", self.sr),
                ("State       ", self.state),
            ]
        ]
        return "\n{}".format(" " * 3 * (depth + 1)).join([str(self)] + lines)
