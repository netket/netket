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

os.environ["JAX_ENABLE_X64"] = "1"


class Jax(AbstractMachine):
    def __init__(self, hilbert, module, dtype=complex, seed=None):
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
            seed: Seed to use to construct `jax.random.PRNGKey` for
                initialisation of parameters. If `None` seed will be chosen
                automatically (which is __not__ synchronized between MPI
                processes so prefer `self.init_random_parameters` if you're
                using multiple MPI tasks)
        """
        super(Jax, self).__init__(hilbert)

        if dtype is not float and dtype is not complex:
            raise TypeError("dtype must be either float or complex")

        self._dtype = dtype
        self._npdtype = _np.complex128 if dtype is complex else _np.float64

        init_fn, self._forward_fn = module
        self._forward_fn = jax.jit(self._forward_fn)

        if seed is None:
            seed = _np.random.randint(2 ** 32 - 2)

        input_shape = (-1, self.n_visible)
        output_shape, self._params = init_fn(jax.random.PRNGKey(seed), input_shape)

        if output_shape != (-1, 1):
            raise ValueError("A valid network must have 1 output.")

        # Computes total number of parameters
        self._npar = sum(
            reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params
        )

        # Computes the Jacobian matrix using backprop
        self._jacobian = jax.jacrev(
            self._forward_fn, holomorphic=(self._dtype is complex)
        )
        self._jacobian = jax.jit(self._jacobian)

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

        J = self._jacobian(self._params, x)
        batch_size = x.shape[0]
        i = 0
        for g in (g.reshape(batch_size, 1, -1) for layer in J for g in layer):
            n = g.shape[2]
            out[:, i : i + n] = g[:, 0, :]
            i += n

        return out

    def vector_jacobian_prod(self, x, vec, out=None):
        vals, f_jvp = jax.vjp(
            self._forward_fn, self._params, x.reshape((-1, x.shape[-1]))
        )

        pout = f_jvp(vec.reshape(vals.shape).conjugate())

        if out is None:
            out = _np.empty((self.n_par), dtype=_np.complex128)

        k = 0
        for layer in pout:
            for pl in layer:
                for p in pl:
                    out[k : k + p.size] = p.reshape(-1).conjugate()
                    k += p.size

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

    @property
    def parameters(self):
        k = 0
        pars = _np.empty(self._npar, dtype=_np.complex128)
        for i, layers in enumerate(self._params):
            for layer in layers:
                pars[k : k + layer.size] = _np.array(
                    layer.reshape(-1), dtype=_np.complex128
                )
                k += layer.size

        return pars

    @parameters.setter
    def parameters(self, p):

        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self.n_par)
            )

        k = 0
        pars = []
        for i, layers in enumerate(self._params):
            lp = []
            for layer in layers:
                if self._dtype is complex:
                    lp.append(
                        jax.numpy.array(p[k : k + layer.size]).reshape(layer.shape)
                    )
                else:
                    lp.append(
                        jax.numpy.array(p[k : k + layer.size].real).reshape(layer.shape)
                    )
                k += layer.size
            pars.append(tuple(lp))

        self._params = pars

        npar = sum(reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params)

        assert npar == self._npar
