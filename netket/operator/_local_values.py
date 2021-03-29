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

from netket.legacy.machine.density_matrix.abstract_density_matrix import (
    AbstractDensityMatrix as DensityMatrix,
)

from ._local_liouvillian import LocalLiouvillian as _LocalLiouvillian


@jit(nopython=True)
def _local_values_kernel(log_vals, log_val_primes, mels, sections, out):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s] * np.exp(log_val_primes[low_range:s] - log_vals[i])
        ).sum()
        low_range = s


def _local_values_impl(op, machine, v, log_vals, out):

    sections = np.empty(v.shape[0], dtype=np.int32)
    v_primes, mels = op.get_conn_flattened(np.asarray(v), sections)

    log_val_primes = machine.log_val(v_primes)

    _local_values_kernel(
        np.asarray(log_vals), np.asarray(log_val_primes), mels, sections, out
    )


@jit(nopython=True)
def _op_op_unpack_kernel(v, sections, vold):

    low_range = 0
    for i, s in enumerate(sections):
        vold[low_range:s] = v[i]
        low_range = s

    return vold


def _local_values_op_op_impl(op, machine, v, log_vals, out):

    sections = np.empty(v.shape[0], dtype=np.int32)
    v_np = np.asarray(v)
    v_primes, mels = op.get_conn_flattened(v_np, sections)

    vold = np.empty((sections[-1], v.shape[1]))
    _op_op_unpack_kernel(v_np, sections, vold)

    log_val_primes = machine.log_val(v_primes, vold)

    _local_values_kernel(
        np.asarray(log_vals), np.asarray(log_val_primes), mels, sections, out
    )


def local_values(op, machine, v, log_vals=None, out=None):
    r"""
    Computes local values of the operator `op` for all `samples`.

    The local value is defined as
    .. math:: O_{\mathrm{loc}}(x) = \langle x | O | \Psi \rangle / \langle x | \Psi \rangle


            Args:
                op: Hermitian operator.
                v: A numpy array or matrix containing either a batch of visible
                    configurations :math:`V = v_1,\dots v_M`.
                    Each row of the matrix corresponds to a visible configuration.
                machine: Wavefunction :math:`\Psi`.
                log_vals: A scalar/numpy array containing the value(s) :math:`\Psi(V)`.
                    If not given, it is computed from scratch.
                    Defaults to None.
                out: A scalar or a numpy array of local values of the operator.
                    If not given, it is allocated from scratch and then returned.
                    Defaults to None.

            Returns:
                If samples is given in batches, a numpy array of local values
                of the operator, otherwise a scalar.
    """

    # True when this is the local_value of a densitymatrix times an operator (observable)
    is_op_times_op = isinstance(machine, DensityMatrix) and not isinstance(
        op, _LocalLiouvillian
    )
    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    if log_vals is None:
        if not is_op_times_op:
            log_vals = machine.log_val(v)
        else:
            log_vals = machine.log_val(v, v)

    if not is_op_times_op:
        _impl = _local_values_impl
    else:
        _impl = _local_values_op_op_impl

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)

    if out is None:
        out = np.empty(v.shape[0], dtype=np.complex128)

    _impl(op, machine, v, log_vals, out)

    return out
