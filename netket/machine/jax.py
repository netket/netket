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


__all__ = ["Jax"]


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
        self._params = []
        if self._dtype is complex:
            output_shape, pars_real = init_fn(jax.random.PRNGKey(seed), input_shape)
            _, pars_imag = init_fn(jax.random.PRNGKey(seed + 1), input_shape)
            if output_shape != (-1, 1):
                raise ValueError("A complex valued network must have only 1 output.")
            for x1, x2 in zip(pars_real, pars_imag):
                layer_state = []
                for l1, l2 in zip(x1, x2):
                    layer_state += [
                        _np.array(l1 + 1j * l2, dtype=self._npdtype),
                    ]
                self._params.append(layer_state)
        else:
            output_shape, pars = init_fn(jax.random.PRNGKey(seed), input_shape)
            if output_shape != (-1, 2):
                raise ValueError("A real valued network must have 2 outputs.")
            for x1 in pars:
                layer_state = []
                for l1 in x1:
                    layer_state += [
                        _np.array(l1, dtype=self._npdtype),
                    ]
                self._params.append(layer_state)

        # Computes total number of parameters
        self._npar = sum(
            reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params
        )

        assert all(
            _np.asarray(p).flags.c_contiguous for layer in self._params for p in layer
        ), "sorry, column major order is not supported (yet)"

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
            out = _np.empty(x.shape[0], dtype=_np.complex128)

        if self._dtype is complex:
            out = _np.array(self._forward_fn(self._params, x).reshape(x.shape[0],))
        else:
            a = _np.asarray(self._forward_fn(self._params, x))
            out[:] = (a[:, 0] + 1j * a[:, 1]).squeeze()
        return out

    def der_log(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        if out is None:
            out = _np.empty((x.shape[0], self.n_par), dtype=_np.complex128)

        J = self._jacobian(self._params, x)
        batch_size = x.shape[0]
        i = 0
        if self._dtype is complex:
            for g in (g.reshape(batch_size, 1, -1) for layer in J for g in layer):
                n = g.shape[2]
                out[:, i : i + n] = g[:, 0, :]
                i += n
        else:
            for g in (g.reshape(batch_size, 2, -1) for layer in J for g in layer):
                n = g.shape[2]
                out[:, i : i + n].real = g[:, 0, :]
                out[:, i : i + n].imag = g[:, 1, :]
                i += n
        return out

    @property
    def is_holomorphic(self):
        return self._dtype is complex

    @property
    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), _np.asarray(p).view()))
        return OrderedDict(state)

    @property
    def parameters(self):
        return _np.concatenate(
            tuple(
                p.astype(dtype=_np.complex128).reshape(-1)
                for p in self.state_dict.values()
            )
        )

    @parameters.setter
    def parameters(self, p):
        if p.shape != (self.n_par,):
            raise ValueError(
                "p has wrong shape: {}; expected ({},)".format(p.shape, self.n_par)
            )

        i = 0
        for x in map(lambda x: x.reshape(-1), self.state_dict.values()):
            if self._dtype is complex:
                _np.copyto(x, p[i : i + x.size])
            else:
                _np.copyto(x, p[i : i + x.size].real)
            i += x.size
