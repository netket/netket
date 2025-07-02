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
from typing import Hashable
from collections.abc import Callable

import numpy as np

import jax
import jax.numpy as jnp

from netket.utils import struct
from netket.utils.types import DType, PyTree, Array
import netket.jax as nkjax
from netket.hilbert import AbstractHilbert
from netket.operator import ContinuousOperator
from netket.utils import HashableArray


def jacrev(f):
    def jacfun(x):
        y, vjp_fun = nkjax.vjp(f, x)
        if y.size == 1:
            eye = jnp.eye(y.size, dtype=x.dtype)[0]
            eye = jax.lax.pvary(eye, tuple(jax.typeof(y).vma))
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        else:
            eye = jnp.eye(y.size, dtype=x.dtype)
            eye = jax.lax.pvary(eye, tuple(jax.typeof(y).vma))
            J = jax.vmap(vjp_fun, in_axes=0)(eye)
        return J

    return jacfun


def jacfwd(f):
    def jacfun(x):
        jvp_fun = lambda s: jax.jvp(f, (x,), (s,))[1]
        eye = jnp.eye(len(x), dtype=x.dtype)
        J = jax.vmap(jvp_fun, in_axes=0)(eye)
        return J

    return jacfun


class KineticEnergy(ContinuousOperator):
    r"""This is the kinetic energy operator (hbar = 1). The local value is given by:
    :math:`E_{kin} = -1/2 ( \sum_i \frac{1}{m_i} (\log(\psi))'^2 + (\log(\psi))'' )`
    """

    _is_hermitian: bool = struct.static_field()
    _mass: Array = struct.field()

    def __init__(
        self,
        hilbert: AbstractHilbert,
        mass: float | list[float],
        dtype: DType | None = None,
    ):
        r"""Args:
        hilbert: The underlying Hilbert space on which the operator is defined
        mass: float if all masses are the same, list indicating the mass of each particle otherwise
        dtype: Data type of the mass
        """

        self._mass = jnp.asarray(mass, dtype=dtype)
        self._is_hermitian = np.allclose(self._mass.imag, 0.0)

        super().__init__(hilbert, self._mass.dtype)

    @property
    def mass(self):
        return self._mass

    @property
    def is_hermitian(self):
        return self._is_hermitian

    def _expect_kernel_single(
        self,
        logpsi: Callable,
        params: PyTree,
        x: Array,
    ):
        def logpsi_x(x):
            return logpsi(params, x)

        dlogpsi_x = jacrev(logpsi_x)
        inverse_mass = jnp.reciprocal(self._mass)

        dp_dx2 = jnp.diag(jacfwd(dlogpsi_x)(x)[0].reshape(x.shape[0], x.shape[0]))
        dp_dx = dlogpsi_x(x)[0][0] ** 2

        return -0.5 * jnp.sum(inverse_mass * (dp_dx2 + dp_dx), axis=-1)

    def _expect_kernel(
        self,
        logpsi: Callable,
        params: PyTree,
        x: Array,
    ):
        return jax.vmap(
            lambda x_: self._expect_kernel_single(logpsi, params, x_), in_axes=(0)
        )(x)

    @struct.property_cached(pytree_ignore=True)
    def _attrs(self) -> tuple[Hashable, ...]:
        return (self.hilbert, self.dtype, HashableArray(self.mass))

    def __repr__(self):
        return f"KineticEnergy(m={self._mass})"
