import jax as jax
import numpy as _np
from numba import jit
from functools import partial

from ._local_liouvillian import LocalLiouvillian as _LocalLiouvillian
from ..vmc_common import tree_map

########################################
# Perform AD through the local values and then vmap.
# Used to compute the gradient
# \sum_i mel(i) * exp(vp(i)-v) * ( O_k(vp(i)) - O_k(v) )

#  Assumes that v is a single state (Vector) and vp is a batch (matrix). pars can be a pytree.
@partial(jax.jit, static_argnums=4)
def _local_value_kernel(pars, vp, mel, v, logpsi):
    return jax.numpy.sum(mel * jax.numpy.exp(logpsi(pars, vp) - logpsi(pars, v)))


#  Assumes that v is a single state (Vector) and vp is a batch (matrix). pars can be a pytree.
_der_local_value_kernel = jax.jit(
    jax.grad(_local_value_kernel, argnums=0, holomorphic=True), static_argnums=4,
)

#  Assumes that v is a batch (matrix) and vp is a batch-batch (3-tensor).
_der_local_values_kernel = jax.jit(
    jax.vmap(_der_local_value_kernel, in_axes=(None, 0, 0, 0, None), out_axes=0),
    static_argnums=4,
)

########################################
# Perform AD through the local values and then vmap.
# Also return the local_value, which is the local_energy.
# unused at the moment, but could be exploited to decrease multiple computations
# of local energy.
#

# same assumptions as above
_local_value_and_grad_kernel = jax.jit(
    jax.value_and_grad(_local_value_kernel, argnums=0, holomorphic=True),
    static_argnums=4,
)

_local_values_and_grads_kernel = jax.jit(
    jax.vmap(
        _local_value_and_grad_kernel, in_axes=(None, 0, 0, 0, None), out_axes=(0, 0)
    ),
    static_argnums=4,
)

########################################
# Computes the non-centered gradient of local values
# \sum_i mel(i) * exp(vp(i)-v) * O_k(i)
@partial(jax.jit, static_argnums=(4, 5))
def _local_value_and_grad_notcentered_kernel(
    pars, vp, mel, v, logpsi, real_to_complex=False
):
    # can use if with jit because that argument is exposed statically to the jit!
    if real_to_complex:
        logpsi_vp_r, f_vjp_r = jax.vjp(lambda w: (logpsi(w, vp).real), pars)
        logpsi_vp_j, f_vjp_j = jax.vjp(lambda w: (logpsi(w, vp).imag), pars)

        logpsi_vp = logpsi_vp_r + 1.0j * logpsi_vp_j

        vec = mel * jax.numpy.exp(logpsi_vp - logpsi(pars, v))
        vec_r = vec.real
        vec_j = vec.imag

        loc_val = vec.sum()

        vr_grad_r, tree_fun = jax.tree_flatten(f_vjp_r(vec_r)[0])
        vj_grad_r, _ = jax.tree_flatten(f_vjp_r(vec_j)[0])
        vr_grad_j, _ = jax.tree_flatten(f_vjp_j(vec_r)[0])
        vj_grad_j, _ = jax.tree_flatten(f_vjp_j(vec_j)[0])

        r_flat = [rr + 1j * jr for rr, jr in zip(vr_grad_r, vj_grad_r)]
        j_flat = [rr + 1j * jr for rr, jr in zip(vr_grad_j, vj_grad_j)]
        out_flat = [re + 1.0j * im for re, im in zip(r_flat, j_flat)]

        grad_c = jax.tree_unflatten(tree_fun, out_flat)
    else:
        logpsi_vp, f_vjp = jax.vjp(lambda w: logpsi(w, vp), pars)

        vec = mel * jax.numpy.exp(logpsi_vp - logpsi(pars, v))

        loc_val = vec.sum()
        grad_c = f_vjp(vec)[0]

    return loc_val, grad_c


@partial(jax.jit, static_argnums=(4, 5))
def _local_values_and_grads_notcentered_kernel(
    pars, vp, mel, v, logpsi, real_to_complex=False
):
    f_vmap = jax.vmap(
        _local_value_and_grad_notcentered_kernel,
        in_axes=(None, 0, 0, 0, None, None),
        out_axes=(0, 0),
    )
    return f_vmap(pars, vp, mel, v, logpsi, real_to_complex)


def _der_local_values_notcentered_impl(op, machine, v, log_vals):
    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v._value, sections, pad=True)

    n_primes = sections[0]
    n_visible = v.shape[1]

    v_primes_r = v_primes.reshape(-1, n_primes, n_visible)
    mels_r = mels.reshape(-1, n_primes)

    if machine._dtype is float and machine._outdtype is complex:
        real_to_complex = True
    else:
        real_to_complex = False

    val, grad = _local_values_and_grads_notcentered_kernel(
        machine.parameters, v_primes_r, mels_r, v, machine.jax_forward, real_to_complex
    )
    return grad


def _der_local_values_impl(op, machine, v, log_vals):
    sections = _np.empty(v.shape[0], dtype=_np.int32)
    v_primes, mels = op.get_conn_flattened(v._value, sections, pad=True)

    n_primes = sections[0]
    n_visible = v.shape[1]

    v_primes_r = v_primes.reshape(-1, n_primes, n_visible)
    mels_r = mels.reshape(-1, n_primes)

    if machine._dtype is complex:
        pars = pars
    else:
        pars = tree_map(lambda v: v.astype(jax.numpy.complex128), machine._params)

    val, grad = _local_values_and_grads_kernel(
        pars, v_primes_r, mels_r, v, machine.jax_forward
    )
    return grad


def der_local_values_jax(
    op,
    machine,
    v,
    log_vals=None,
    # der_log_vals=None,
    center_derivative=True,
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
    if v.ndim != 2:
        raise RuntimeError("Invalid input shape, expected a 2d array")

    assert (
        v.shape[1] == op.hilbert.size
    ), "samples has wrong shape: {}; expected (?, {})".format(v.shape, op.hilbert.size)

    if log_vals is None:
        log_vals = machine.log_val(v)

    if center_derivative is True:
        return _der_local_values_impl(op, machine, v, log_vals,)
    else:
        return _der_local_values_notcentered_impl(op, machine, v, log_vals,)
