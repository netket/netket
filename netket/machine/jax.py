# Copyright 2019 The Simons Foundation, Inc. - All Rights Reserved.
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

import jax
from collections import OrderedDict
from functools import reduce

from .abstract_machine import AbstractMachine

import numpy as _np
from netket.random import randint as _randint
from jax.tree_util import tree_flatten, tree_unflatten, tree_map


class Jax(AbstractMachine):
    def __init__(self, hilbert, module, dtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet machine.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the network must have a single output.
        """
        super().__init__(hilbert)

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        self._init_fn, self._forward_fn = module

        # Computes the Jacobian matrix using forward ad
        self._forward_fn = jax.jit(self._forward_fn)

        forward_scalar = jax.jit(lambda pars, x: self._forward_fn(pars, x).reshape(()))
        grad_fun = jax.jit(jax.grad(forward_scalar, holomorphic=self.is_holomorphic))
        self._perex_grads = jax.jit(jax.vmap(grad_fun, in_axes=(None, 0)))

        self.init_random_parameters()

        # Computes total number of parameters
        weights, _ = tree_flatten(self._params)
        self._npar = sum([w.size for w in weights])
        
    def init_random_parameters(self, seed=None, sigma=None):
        if seed is None:
            seed = _randint(0, 2 ** 32 - 2)

        input_shape = (-1, self.input_size)
        output_shape, params = self._init_fn(jax.random.PRNGKey(seed), input_shape)

        self._params = self._cast(params)

        if output_shape != (-1, 1):
            raise ValueError("A valid network must have 1 output.")

    def _cast(self, p):
        if self._dtype is complex:
            from jax.tree_util import tree_unflatten, tree_flatten

            # TODO use tree_map instead
            value_flat, value_tree = tree_flatten(p)
            values_c = list(map(lambda v: v.astype(jax.numpy.complex128), value_flat))

            return tree_unflatten(value_tree, values_c)
        else:
            return p

    @property
    def n_par(self):
        r"""The number of variational parameters in the machine."""
        return self._npar

    def log_val(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.array(
                self._forward_fn(self._params, x).reshape(x.shape[0],),
                dtype=_np.complex128,
            )
        else:
            out[:] = _np.array(
                self._forward_fn(self._params, x).reshape(x.shape[0],),
                dtype=_np.complex128,
            )
        return out

    @property
    def jax_forward(self):
        return self._forward_fn

    def der_log(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        # Jax has bugs for R->C functions...
        out = self._perex_grads(self._params_ascomplex, x)

        return out

    def vector_jacobian_prod(
        self, x, vec, out=None, conjugate=True, return_jacobian=False
    ):
        r"""Computes the scalar product between gradient of the logarithm of the wavefunction for a
        batch of visible configurations `x` and a vector `vec`. The result is stored into `out`.

        Args:
             x: a matrix of `float64` of shape `(*, self.n_visible)`.
             vec: a `complex128` vector used to compute the inner product with the jacobian.
             out: The result of the inner product, it is a vector of `complex128` and length `self.n_par`.
             conjugate (bool): If true, this computes the conjugate of the vector jacobian product.
             return_jacobian (bool): If true, the Jacobian is explicitely computed and returned.


        Returns:
             `out` only or (out,jacobian) if return_jacobian is True
        """
        if not return_jacobian:
            vals, f_jvp = jax.vjp(
                self._forward_fn, self._params, x.reshape((-1, x.shape[-1]))
            )
            pout = f_jvp(vec.reshape(vals.shape).conjugate())
            if conjugate and self._dtype is complex:
                out = tree_map(jax.numpy.conjugate, pout[0])
            else:
                out = pout

            return out

        else:

            if conjugate and self._dtype is complex:
                prodj = lambda j: jax.numpy.tensordot(
                    vec.transpose(), j.conjugate(), axes=1
                )
            else:
                prodj = lambda j: jax.numpy.tensordot(
                    vec.transpose().conjugate(), j, axes=1
                )

            jacobian = self._perex_grads(self._params, x)
            out = tree_map(prodj, jacobian)

            return out, jacobian

    @property
    def is_holomorphic(self):
        return True

    @property
    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), p))
        return OrderedDict(state)

    @property
    def parameters(self):
        return self._params

    @property
    def _params_ascomplex(self):
        if self._dtype is not complex:
            return tree_map(lambda v: v.astype(jax.numpy.complex128), self._params)
        else:
            return self._params

    @parameters.setter
    def parameters(self, p):
        self._params = p
        weights, _ = tree_flatten(self._params)
        npar = sum([w.size for w in weights])

        assert npar == self._npar

    def numpy_flatten(self, data):
        r"""Returns a flattened numpy array representing the given data.
            This is typically used to serialize parameters and gradients.

        Args:
             data: a (possibly non-flat) structure containing jax arrays.

        Returns:
             numpy.ndarray: a one-dimensional array containing a copy of data
        """

        return _np.concatenate(tuple(fd.reshape(-1) for fd in tree_flatten(data)[0]))

    def numpy_unflatten(self, data, shape_like):
        r"""Attempts a deserialization of the given numpy data.
            This is typically used to deserialize parameters and gradients.

        Args:
             data: a 1d numpy array.
             shape_like: this as in instance having the same type and shape of
                         the desired conversion.

        Returns:
             A possibly non-flat structure of jax arrays containing a copy of data
             compatible with the given shape.
        """
        shf, tree = tree_flatten(shape_like)

        datalist = []
        k = 0
        for s in shf:
            size = s.size
            datalist.append(jax.numpy.asarray(data[k : k + size]).reshape(s.shape))
            k += size

        return tree_unflatten(tree, datalist)


from jax.experimental import stax
from jax.experimental.stax import Dense


def SumLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs.sum(axis=-1)

    return init_fun, apply_fun


@jax.jit
def logcosh(x):
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)


LogCoshLayer = stax.elementwise(logcosh)


def JaxRbm(hilbert, alpha, dtype=complex):
    return Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer()),
        dtype=dtype,
    )
