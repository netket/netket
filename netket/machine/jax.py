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
import os
from .abstract_machine import AbstractMachine

import numpy as _np
from netket.random import randint as _randint
from jax.tree_util import tree_flatten, tree_unflatten
from jax.util import safe_map
from netket.stats import sum_inplace as _sum_inplace
from netket.utils import node_number

os.environ["JAX_ENABLE_X64"] = "1"


class Jax(AbstractMachine):
    def __init__(self, hilbert, module, dtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet machine.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                If dtype is float, the network should have 2 outputs corresponding
                to the real and imaginary part of log(psi(x)).
                If dtype is complex, the network should have only 1 output
                representing the complex amplitude log(psi(x)).
        """
        super(Jax, self).__init__(hilbert)

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        self._init_fn, self._forward_fn = module

        # Computes the Jacobian matrix using forward ad
        grad_fun = jax.jit(jax.grad(self._forward_fn, holomorphic=self.is_holomorphic))
        self._forward_fn = jax.jit(self._forward_fn)
        self._perex_grads = jax.jit(jax.vmap(grad_fun, in_axes=(None, 0)))

        self.init_random_parameters()

        # Computes total number of parameters
        self._npar = sum(
            reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params
        )

    def init_random_parameters(self, seed=None, sigma=None):
        if seed is None:
            seed = _randint(0, 2 ** 32 - 2)

        input_shape = (-1, self.n_visible)
        output_shape, params = self._init_fn(jax.random.PRNGKey(seed), input_shape)

        self._params = self._cast(params)

        if output_shape != (-1, 1):
            raise ValueError("A valid network must have 1 output.")

    def _cast(self, p):
        if self._dtype is complex:
            from jax.tree_util import tree_unflatten, tree_flatten

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

    @property
    def jax_parameters(self):
        return self._params

    def der_log(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        J = self._perex_grads(self._params, x)

        return self._convert_jacobian(J, out, x.shape[0])

    def _convert_jacobian(self, J, out, bsize):
        k = 0
        for layer in J:
            for pl in layer:
                pl = pl.reshape((bsize, -1))
                out[:, k : k + pl.shape[1]] = pl
                k += pl.shape[1]
        return out

    def vector_jacobian_prod(self, x, vec, out=None, distributed=True):
        vals, f_jvp = jax.vjp(
            self._forward_fn, self._params, x.reshape((-1, x.shape[-1]))
        )

        pout = f_jvp(vec.reshape(vals.shape).conjugate())
        out = pout[0]

        if distributed:
            flat_out, tree = tree_flatten(out)
            # converting to numpy before the reduction
            # this copy is unavoidable because jax types are not writable
            # when viewed as numpy types
            np_arr = list(map(_np.array, flat_out))

            # sum reduction in place
            safe_map(_sum_inplace, np_arr)

            # back to jax types
            flat_out = list(map(jax.numpy.array, np_arr))
            out = tree_unflatten(tree, flat_out)

        return out

    @property
    def is_holomorphic(self):
        return self._dtype is complex

    @property
    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), p))
        return OrderedDict(state)

    def save(self, file):
        assert type(file) is str
        with open(file, "wb") as file_ob:
            jax.numpy.save(file_ob, self.parameters, allow_pickle=True)

    def load(self, file):
        self.parameters = jax.numpy.load(file, allow_pickle=True)

    @property
    def parameters(self):
        return self._params

    @parameters.setter
    def parameters(self, p):
        self._params = p
        npar = sum(reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params)

        assert npar == self._npar


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
