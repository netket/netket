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

from typing import Union
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.operator import AbstractOperator
from netket.hilbert import AbstractHilbert
from netket.vqs import FullSumState
from netket.utils import mpi
from netket.utils.types import Array
from netket.utils.dispatch import dispatch

BaseType = Union[AbstractOperator, np.ndarray, str]


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
    return jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


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
    Same as _avg_O, but for FullSumState.
    """
    sigma = hilbert.all_states()
    sigma = sigma.reshape((-1, sigma.shape[-1]))
    _, vjp = nkjax.vjp(lambda W: afun({"params": W, **model_state}, sigma), pars)
    psi_2 = jnp.abs(jnp.exp(afun({"params": pars, **model_state}, sigma))) ** 2
    psi_2 /= jnp.sum(psi_2)
    (O_avg,) = vjp(psi_2)
    return jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)


@dispatch
def _grad_negative(state_diag: FullSumState):  # noqa: F811
    r"""
    Same as _grad_negative, but for FullSumState.
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

    O_avg = jax.tree_util.tree_map(lambda x: mpi.mpi_mean_jax(x)[0], O_avg)

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
