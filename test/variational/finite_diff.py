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

import jax
import jax.numpy as jnp

import netket as nk


def expval(par, vstate, H, real=False):
    vstate.parameters = par
    psi = vstate.to_array()
    expval = psi.conj() @ (H @ psi)
    if real:
        expval = np.real(expval)

    return expval


def central_diff_grad(func, x, eps, *args, dtype=None):
    if dtype is None:
        dtype = x.dtype

    grad = np.zeros(
        len(x), dtype=nk.jax.maybe_promote_to_complex(x.dtype, func(x, *args).dtype)
    )
    epsd = np.zeros(len(x), dtype=dtype)
    epsd[0] = eps
    for i in range(len(x)):
        assert not np.any(np.isnan(x + epsd))
        grad_r = 0.5 * (func(x + epsd, *args) - func(x - epsd, *args))
        if nk.jax.is_complex(x):
            grad_i = 0.5 * (func(x + 1j * epsd, *args) - func(x - 1j * epsd, *args))
            grad[i] = 0.5 * grad_r + 0.5j * grad_i
        else:
            # grad_i = 0.0
            grad[i] = grad_r

        assert not np.isnan(grad[i])
        grad[i] /= eps
        epsd = np.roll(epsd, 1)
    return grad


def same_derivatives(der, num_der, abs_eps=1.0e-6, rel_eps=1.0e-6):
    """
    Checks that two complex-valued arrays are the same.

    Same as `np.testing.assert_allclose` but checks the real
    and imaginary parts independently for better error reporting.
    """
    assert der.shape == num_der.shape

    np.testing.assert_allclose(der.real, num_der.real, rtol=rel_eps, atol=abs_eps)

    np.testing.assert_allclose(der.imag, num_der.imag, rtol=rel_eps, atol=abs_eps)


def same_log_derivatives(der_log, num_der_log, abs_eps=1.0e-6, rel_eps=1.0e-6):
    """
    Checks that two log-derivatives are equivalent
    """
    assert der_log.shape == num_der_log.shape

    np.testing.assert_allclose(
        der_log.real, num_der_log.real, rtol=rel_eps, atol=abs_eps
    )

    # compute the distance between the two phases modulo 2pi
    delta_der_log = der_log.imag - num_der_log.imag

    # the distance is taken to be the minimum of the distance between
    # (|A-B|mod 2Pi) and (|B-A| mod 2Pi)
    delta_der_log_mod_1 = np.mod(delta_der_log, 2 * np.pi)
    delta_der_log_mod_2 = np.mod(-delta_der_log, 2 * np.pi)
    delta_der_log = np.minimum(delta_der_log_mod_1, delta_der_log_mod_2)

    # Compare against pi and not 0 because otherwise rtol will fail always
    np.testing.assert_allclose(
        delta_der_log + np.pi,
        np.pi,
        rtol=rel_eps,
        atol=abs_eps,
    )
