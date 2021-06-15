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

import numpy as np
from numba import jit

from netket.legacy.machine import Jax as _Jax

from ._der_local_values_jax import der_local_values_jax


@jit(nopython=True)
def _der_local_values_kernel(
    log_vals, log_val_p, mels, der_log, der_log_p, sections, out
):
    low_range = 0
    for i, s in enumerate(sections):
        out[i, :] = (
            np.expand_dims(
                mels[low_range:s] * np.exp(log_val_p[low_range:s] - log_vals[i]), 1
            )
            * (der_log_p[low_range:s, :] - der_log[i, :])
        ).sum(axis=0)
        low_range = s


def _der_local_values_impl(op, machine, v, log_vals, der_log_vals, out, batch_size=64):
    sections = np.empty(v.shape[0], dtype=np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    log_val_primes = machine.log_val(v_primes)

    # Compute the der_log in small batches and not in one go.
    # For C++ machines there is a 100% slowdown when the batch is too big.
    n_primes = len(log_val_primes)
    der_log_primes = np.empty((n_primes, machine.n_par), dtype=np.complex128)

    for s in range(0, n_primes, batch_size):
        end = min(s + batch_size, n_primes)
        der_log_primes[s:end, :] = machine.der_log(v_primes[s:end])

    _der_local_values_kernel(
        log_vals, log_val_primes, mels, der_log_vals, der_log_primes, sections, out
    )


@jit(nopython=True)
def _der_local_values_notcentered_kernel(
    log_vals, log_val_p, mels, der_log_p, sections, out
):
    low_range = 0
    for i, s in enumerate(sections):
        out[i, :] = (
            np.expand_dims(
                mels[low_range:s] * np.exp(log_val_p[low_range:s] - log_vals[i]), 1
            )
            * der_log_p[low_range:s, :]
        ).sum(axis=0)
        low_range = s


def _der_local_values_notcentered_impl(op, machine, v, log_vals, out, batch_size=64):
    sections = np.empty(v.shape[0], dtype=np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    log_val_primes = machine.log_val(v_primes)

    # Compute the der_log in small batches and not in one go.
    # For C++ machines there is a 100% slowdown when the batch is too big.
    n_primes = len(log_val_primes)
    der_log_primes = np.empty((n_primes, machine.n_par), dtype=np.complex128)

    for s in range(0, n_primes, batch_size):
        end = min(s + batch_size, n_primes)
        der_log_primes[s:end, :] = machine.der_log(v_primes[s:end])

    _der_local_values_notcentered_kernel(
        log_vals, log_val_primes, mels, der_log_primes, sections, out
    )


def der_local_values(
    op,
    machine,
    v,
    log_vals=None,
    der_log_vals=None,
    out=None,
    center_derivative=True,
    batch_size=64,
):
    r"""
    Computes the derivative of local values of the operator `op` for all `samples`.

    The local value is defined as
    .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle


            Args:
                op: Hermitian operator.
                v: A numpy array or matrix containing either a single
                    :math:`V = v` or a batch of visible
                    configurations :math:`V = v_1,\dots v_M`.
                    In the latter case, each row of the matrix corresponds to a
                    visible configuration.
                machine: Wavefunction :math:`\Psi`.
                log_vals: A scalar/numpy array containing the value(s) :math:`\Psi(V)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                der_log_vals: A numpy tensor containing the vector of log-derivative(s) :math:`O_i(V)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                out: A scalar or a numpy array of local values of the operator.
                    If not given, it is allocated from scratch and then returned.
                    Defaults to None.
                center_derivative: Whever to center the derivatives or not. In the formula above,
                    When this is true/false it is equivalent to setting :math:`\alpha=\{1 / 2\}`.
                    By default `center_derivative=True`, meaning that it returns the correct
                    derivative of the local values. False is mainly used when dealing with liouvillians.

            Returns:
                If samples is given in batches, a numpy ndarray of derivatives of local values
                of the operator, otherwise a 1D array.
    """
    if isinstance(machine, _Jax):
        return der_local_values_jax(
            op,
            machine,
            v,
            log_vals=log_vals,
            center_derivative=center_derivative,
        )

    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)

    if out is None:
        out = np.empty((v.shape[0], machine.n_par), dtype=np.complex128)

    if log_vals is None:
        log_vals = machine.log_val(v)

    if der_log_vals is None and center_derivative is True:
        der_log_vals = machine.der_log(v)

    if center_derivative is True:
        _der_local_values_impl(
            op, machine, v, log_vals, der_log_vals, out, batch_size=batch_size
        )
    else:
        _der_local_values_notcentered_impl(
            op, machine, v, log_vals, out, batch_size=batch_size
        )

    return out
