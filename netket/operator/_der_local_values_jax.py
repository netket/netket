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
from jax import numpy as jnp

from netket.legacy.machine._jax_utils import (
    outdtype,
    outdtype_iscomplex,
    tree_leaf_iscomplex,
)

from ._local_liouvillian import LocalLiouvillian as _LocalLiouvillian
from ._local_cost_functions import (
    define_local_cost_function,
    local_costs_and_grads_function,
    local_value_cost,
    local_value_op_op_cost,
)

local_energy_kernel = local_value_cost


########################################
# Perform AD through the local values and then vmap.
# Used to compute the gradient
# \sum_i mel(i) * exp(vp(i)-v) * ( O_k(vp(i)) - O_k(v) )
def _der_local_values_impl(op, machine, v, log_vals):
    v_primes, mels = op.get_conn_padded(np.asarray(v))

    val, grad = local_costs_and_grads_function(
        local_energy_kernel, machine.jax_forward, machine.parameters, v_primes, mels, v
    )
    return grad


########################################
# Computes the non-centered gradient of local values
# \sum_i mel(i) * exp(vp(i)-v) * O_k(i)
@partial(jax.jit, static_argnums=0)
def _local_value_and_grad_notcentered_kernel(logpsi, pars, vp, mel, v):
    import netket.jax as nkjax

    logpsi_vp, f_vjp = nkjax.vjp(lambda w: logpsi(w, vp), pars, conjugate=False)
    vec = mel * jax.numpy.exp(logpsi_vp - logpsi(pars, v))

    # TODO : here someone must bring order to those multiple conjs
    odtype = outdtype(logpsi, pars, v)
    vec = jnp.asarray(jnp.conjugate(vec), dtype=odtype)
    loc_val = vec.sum()
    grad_c = f_vjp(vec.conj())[0]

    return loc_val, grad_c


#    odtype = outdtype(logpsi, pars, v)
#    # can use if with jit because that argument is exposed statically to the jit!
#    # if real_to_complex:
#    if not tree_leaf_iscomplex(pars) and jnp.issubdtype(odtype, jnp.complexfloating):
#        logpsi_vp_r, f_vjp_r = jax.vjp(lambda w: (logpsi(w, vp).real), pars)
#        logpsi_vp_j, f_vjp_j = jax.vjp(lambda w: (logpsi(w, vp).imag), pars)
#
#        logpsi_vp = logpsi_vp_r + 1.0j * logpsi_vp_j
#
#        vec = mel * jax.numpy.exp(logpsi_vp - logpsi(pars, v))
#        vec = jnp.asarray(vec, dtype=odtype)
#
#        vec_r = vec.real
#        vec_j = vec.imag
#
#        loc_val = vec.sum()
#
#        vr_grad_r, tree_fun = jax.tree_flatten(f_vjp_r(vec_r)[0])
#        vj_grad_r, _ = jax.tree_flatten(f_vjp_r(vec_j)[0])
#        vr_grad_j, _ = jax.tree_flatten(f_vjp_j(vec_r)[0])
#        vj_grad_j, _ = jax.tree_flatten(f_vjp_j(vec_j)[0])
#
#        r_flat = [rr + 1j * jr for rr, jr in zip(vr_grad_r, vj_grad_r)]
#        j_flat = [rr + 1j * jr for rr, jr in zip(vr_grad_j, vj_grad_j)]
#        out_flat = [re + 1.0j * im for re, im in zip(r_flat, j_flat)]
#
#        grad_c = jax.tree_unflatten(tree_fun, out_flat)
#    else:
#        logpsi_vp, f_vjp = jax.vjp(lambda w: logpsi(w, vp), pars)
#
#        vec = mel * jax.numpy.exp(logpsi_vp - logpsi(pars, v))
#        # vec = jnp.asarray(vec, dtype=odtype)
#
#        loc_val = vec.sum()
#        grad_c = f_vjp(vec)[0]
#
#    return loc_val, grad_c


@partial(jax.jit, static_argnums=0)
def _local_values_and_grads_notcentered_kernel(logpsi, pars, vp, mel, v):
    f_vmap = jax.vmap(
        _local_value_and_grad_notcentered_kernel,
        in_axes=(None, None, 0, 0, 0),
        out_axes=(0, 0),
    )
    return f_vmap(logpsi, pars, vp, mel, v)


def _der_local_values_notcentered_impl(op, machine, v, log_vals):
    v_primes, mels = op.get_conn_padded(np.asarray(v))

    val, grad = _local_values_and_grads_notcentered_kernel(
        machine.jax_forward, machine.parameters, v_primes, mels, v
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
        return _der_local_values_impl(
            op,
            machine,
            v,
            log_vals,
        )
    else:
        return _der_local_values_notcentered_impl(
            op,
            machine,
            v,
            log_vals,
        )
