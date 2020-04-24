import numpy as _np
from numba import jit

from .local_liouvillian import LocalLiouvillian as _LocalLiouvillian
from .._C_netket.machine import DensityMatrix


@jit(nopython=True)
def _local_values_kernel(log_vals, log_val_primes, mels, sections, out):
    low_range = 0
    for i, s in enumerate(sections):
        out[i] = (
            mels[low_range:s] * _np.exp(log_val_primes[low_range:s] - log_vals[i])
        ).sum()
        low_range = s


def _local_values_impl(op, machine, v, log_vals, out):

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    log_val_primes = machine.log_val(v_primes)

    _local_values_kernel(log_vals, log_val_primes, mels, sections, out)


@jit(nopython=True)
def _op_op_unpack_kernel(v, sections, vold):

    low_range = 0
    for i, s in enumerate(sections):
        vold[low_range:s] = v[i]
        low_range = s

    return vold


def _local_values_op_op_impl(op, machine, v, log_vals, out):

    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    vold = _np.empty((sections[-1], v.shape[1]))
    _op_op_unpack_kernel(v, sections, vold)

    log_val_primes = machine.log_val(v_primes, vold)

    _local_values_kernel(log_vals, log_val_primes, mels, sections, out)


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
        out = _np.empty(v.shape[0], dtype=_np.complex128)

    _impl(op, machine, v, log_vals, out)

    return out


@jit(nopython=True)
def _der_local_values_kernel(
    log_vals, log_val_p, mels, der_log, der_log_p, sections, out
):
    low_range = 0
    for i, s in enumerate(sections):
        out[i, :] = (
            _np.expand_dims(
                mels[low_range:s] * _np.exp(log_val_p[low_range:s] - log_vals[i]), 1
            )
            * (der_log_p[low_range:s, :] - der_log[i, :])
        ).sum(axis=0)
        low_range = s


def _der_local_values_impl(op, machine, v, log_vals, der_log_vals, out, batch_size=64):
    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    log_val_primes = machine.log_val(v_primes)

    # Compute the der_log in small batches and not in one go.
    # For C++ machines there is a 100% slowdown when the batch is too big.
    n_primes = len(log_val_primes)
    der_log_primes = _np.empty((n_primes, machine.n_par), dtype=_np.complex128)

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
            _np.expand_dims(
                mels[low_range:s] * _np.exp(log_val_p[low_range:s] - log_vals[i]), 1
            )
            * der_log_p[low_range:s, :]
        ).sum(axis=0)
        low_range = s


def _der_local_values_notcentered_impl(op, machine, v, log_vals, out, batch_size=64):
    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v, sections)

    log_val_primes = machine.log_val(v_primes)

    # Compute the der_log in small batches and not in one go.
    # For C++ machines there is a 100% slowdown when the batch is too big.
    n_primes = len(log_val_primes)
    der_log_primes = _np.empty((n_primes, machine.n_par), dtype=_np.complex128)

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
    if v.ndim == 3:
        assert (
            v.shape[2] == op.hilbert.size
        ), "samples has wrong shape: {}; expected (?, {})".format(
            v.shape, op.hilbert.size
        )

        if out is None:
            out = _np.empty(
                (v.shape[0] * v.shape[1], machine.n_par), dtype=_np.complex128
            )

        if log_vals is None:
            log_vals = machine.log_val(v)

        if der_log_vals is None and center_derivative is True:
            der_log_vals = machine.der_log(v)

        if center_derivative is True:
            _der_local_values_impl(
                op,
                machine,
                v.reshape(-1, op.hilbert.size),
                log_vals.reshape(-1),
                der_log_vals.reshape(-1, machine.n_par),
                out.reshape(-1, machine.n_par),
                batch_size=batch_size,
            )
        else:
            _der_local_values_notcentered_impl(
                op,
                machine,
                v.reshape(-1, op.hilbert.size),
                log_vals.reshape(-1),
                out.reshape(-1, machine.n_par),
                batch_size=batch_size,
            )

        return out.reshape(v.shape[0], v.shape[1], machine.n_par)
    elif v.ndim == 2:
        assert (
            v.shape[1] == op.hilbert.size
        ), "samples has wrong shape: {}; expected (?, {})".format(
            v.shape, op.hilbert.size
        )

        if out is None:
            out = _np.empty((v.shape[0], machine.n_par), dtype=_np.complex128)

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
    elif v.ndim == 1:
        assert v.size == op.hilbert.size, "v has wrong size: {}; expected {}".format(
            v.shape, op.hilbert.size
        )
        if out is None:
            out = _np.empty((1, machine.n_par), dtype=_np.complex128)
        else:
            out = _np.atleast_2d(out)

        if log_vals is None:
            log_vals = machine.log_val(v)

        log_vals = _np.atleast_1d(log_vals)

        if der_log_vals is None and center_derivative is True:
            der_log_vals = machine.der_log(v)

        der_log_vals = _np.atleast_2d(der_log_vals)

        if center_derivative is True:
            _der_local_values_impl(
                op,
                machine,
                v.reshape(1, -1),
                log_vals.reshape(1, -1),
                der_log_vals,
                out,
                batch_size=batch_size,
            )
        else:
            _der_local_values_notcentered_impl(
                op,
                machine,
                v.reshape(1, -1),
                log_vals.reshape(1, -1),
                out,
                batch_size=batch_size,
            )

        return out[0, :]
    raise ValueError(
        "v has wrong dimension: {}; expected either 1, 2 or 3".format(v.ndim)
    )
