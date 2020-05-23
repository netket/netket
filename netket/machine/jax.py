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
    def __init__(self, hilbert, module, dtype=complex, outdtype=None):
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
        super().__init__(hilbert=hilbert, dtype=dtype, outdtype=outdtype)

        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        self._init_fn, self._forward_fn = module

        self._forward_fn_nj = self._forward_fn

        # Computes the Jacobian matrix using forward ad
        self._forward_fn = jax.jit(self._forward_fn)

        forward_scalar = jax.jit(lambda pars, x: self._forward_fn(pars, x).reshape(()))

        # C-> C
        if self._dtype is complex and self._outdtype is complex:

            grad_fun = jax.jit(jax.grad(forward_scalar, holomorphic=True))
            self._perex_grads = jax.jit(jax.vmap(grad_fun, in_axes=(None, 0)))

            def _vjp_fun(pars, v, vec, conjugate, forward_fun):
                vals, f_jvp = jax.vjp(forward_fun, pars, v.reshape((-1, v.shape[-1])))

                out = f_jvp(vec.reshape(vals.shape).conjugate())[0]

                if conjugate:
                    out = tree_map(jax.numpy.conjugate, out)

                return out

            self._vjp_fun = jax.jit(_vjp_fun, static_argnums=(3, 4))

        # R->R
        elif self._dtype is float and self._outdtype is float:

            grad_fun = jax.jit(jax.grad(forward_scalar))
            self._perex_grads = jax.jit(jax.vmap(grad_fun, in_axes=(None, 0)))

            def _vjp_fun(pars, v, vec, conjugate, forward_fun):
                vals, f_jvp = jax.vjp(forward_fun, pars, v.reshape((-1, v.shape[-1])))

                out_r = f_jvp(vec.reshape(vals.shape).real)[0]
                out_i = f_jvp(-vec.reshape(vals.shape).imag)[0]

                r_flat, tree_fun = tree_flatten(out_r)
                i_flat, _ = tree_flatten(out_i)

                if conjugate:
                    out_flat = [re - 1j * im for re, im in zip(r_flat, i_flat)]
                else:
                    out_flat = [re + 1j * im for re, im in zip(r_flat, i_flat)]

                return tree_unflatten(tree_fun, out_flat)

            self._vjp_fun = jax.jit(_vjp_fun, static_argnums=(3, 4))

        # R->C
        elif self._dtype is float and self._outdtype is complex:

            def _gradfun(pars, v):
                grad_r = jax.grad(lambda pars, v: forward_scalar(pars, v).real)(pars, v)
                grad_j = jax.grad(lambda pars, v: forward_scalar(pars, v).imag)(pars, v)

                r_flat, r_fun = tree_flatten(grad_r)
                j_flat, j_fun = tree_flatten(grad_j)

                grad_flat = [re + 1j * im for re, im in zip(r_flat, j_flat)]
                return tree_unflatten(r_fun, grad_flat)

            grad_fun = jax.jit(_gradfun)
            self._perex_grads = jax.jit(jax.vmap(grad_fun, in_axes=(None, 0)))

            def _vjp_fun(pars, v, vec, conjugate, forward_fun):
                v = v.reshape((-1, v.shape[-1]))
                vals_r, f_jvp_r = jax.vjp(
                    lambda pars, v: forward_fun(pars, v).real, pars, v
                )

                vals_j, f_jvp_j = jax.vjp(
                    lambda pars, v: forward_fun(pars, v).imag, pars, v
                )
                vec_r = vec.reshape(vals_r.shape).real
                vec_j = vec.reshape(vals_r.shape).imag

                # val = vals_r + vals_j
                vr_jr = f_jvp_r(vec_r)[0]
                vj_jr = f_jvp_r(vec_j)[0]
                vr_jj = f_jvp_j(vec_r)[0]
                vj_jj = f_jvp_j(vec_j)[0]

                rjr_flat, tree_fun = tree_flatten(vr_jr)
                jjr_flat, _ = tree_flatten(vj_jr)
                rjj_flat, _ = tree_flatten(vr_jj)
                jjj_flat, _ = tree_flatten(vj_jj)

                r_flat = [rr - 1j * jr for rr, jr in zip(rjr_flat, jjr_flat)]
                j_flat = [rr - 1j * jr for rr, jr in zip(rjj_flat, jjj_flat)]
                out_flat = [re + 1j * im for re, im in zip(r_flat, j_flat)]
                if conjugate:
                    out_flat = [x.conjugate() for x in out_flat]

                return tree_unflatten(tree_fun, out_flat)

            self._vjp_fun = jax.jit(_vjp_fun, static_argnums=(3, 4))

        else:
            raise ValueError("We do not support C->R wavefunctions.")

        self.jax_init_parameters()

        # Computes total number of parameters
        weights, _ = tree_flatten(self._params)
        self._npar = sum([w.size for w in weights])
        self.init_random_parameters()

    def jax_init_parameters(self, seed=None):
        """
        Uses the init function of the jax networks to generate a set of parameters.
        Beware that usually jax does not correctly set the imaginary part of
        networks, so for complex networks it MUST be followed by a call to
        init_random_parameters, unless your network correctly initialised the
        imaginary part.
        """
        if seed is None:
            seed = _randint(0, 2 ** 32 - 2)

        input_shape = (-1, self.input_size)
        output_shape, params = self._init_fn(jax.random.PRNGKey(seed), input_shape)

        self._params = self._cast(params)

        if output_shape != (-1, 1):
            raise ValueError("A valid network must have 1 output.")

    def init_random_parameters(self, seed=None, sigma=0.01):
        rgen = _np.random.RandomState(seed)

        pars = rgen.normal(scale=sigma, size=self.n_par)

        if self.has_complex_parameters:
            pars = pars + 1.0j * rgen.normal(scale=sigma, size=self.n_par)

        self.parameters = self.numpy_unflatten(pars, self.parameters)

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
            out = self._forward_fn(self._params, x).reshape(x.shape[0],)
        else:
            out[:] = self._forward_fn(self._params, x).reshape(x.shape[0],)
        return out

    @property
    def jax_forward(self):
        return self._forward_fn

    def der_log(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        # Jax has bugs for R->C functions...
        out = self._perex_grads(self._params, x)

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
            return self._vjp_fun(self._params, x, vec, conjugate, self._forward_fn)

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
    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), p))
        return OrderedDict(state)

    @property
    def parameters(self):
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


def FanInSum2ModPhase():
    """Layer construction function for a fan-in sum layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0]
        return output_shape, tuple()

    def apply_fun(params, inputs, **kwargs):
        output = 1.0 * inputs[0] + 1.0j * inputs[1]
        return output

    return init_fun, apply_fun


FanInSum2ModPhase = FanInSum2ModPhase()


def JaxRbmSpinPhase(hilbert, alpha, dtype=float):
    return Jax(
        hilbert,
        stax.serial(
            stax.FanOut(2),
            stax.parallel(
                stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer()),
                stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer()),
            ),
            FanInSum2ModPhase,
        ),
        dtype=dtype,
        outdtype=complex,
    )
