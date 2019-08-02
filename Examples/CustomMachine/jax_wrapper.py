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

from collections import OrderedDict
from functools import reduce
from pickle import dump, load
import random
import numpy as _np
import jax as _jax

import netket

__all__ = ["JAXMachine"]


class JAXMachine(netket.machine.CxxMachine):
    def __init__(self, hilbert, module, seed=None):
        # NOTE: The following call to __init__ is important!
        super(JAXMachine, self).__init__(hilbert)
        if seed is None:
            # NOTE(twesterhout): I believe, jax uses 32-bit integers internally
            # to represent the seed. Hence the limit
            seed = random.randint(0, 2 ** 32 - 1)
        init_fn, self._forward_fn = module
        # Chosen randomly. It'll change if we forward propagate `v` with a
        # different leading dimension.
        batch_size = 64
        input_shape = (batch_size, hilbert.size)
        output_shape, self._params = init_fn(_jax.random.PRNGKey(seed), input_shape)
        if output_shape != (batch_size, 2):
            raise ValueError("module's output_shape is weird, check your network")
        # Computes total number of parameters
        n_par = sum(reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params)
        # Save the initial dtype (for debugging mostly)
        dtypes = set(p.dtype for layer in self._params for p in layer)
        if len(dtypes) != 1:
            raise ValueError("module parameters have different dtypes")
        self._dtype = next(iter(dtypes))
        self._complex_dtype = {
            _np.dtype('float32'): _np.complex64,
            _np.dtype('float64'): _np.complex128,
        }[self._dtype]
        assert all(
            _np.asarray(p).flags.c_contiguous for layer in self._params for p in layer
        ), "sorry, column major order is not supported (yet)"
        self._n_par = lambda: n_par
        self._jacobian = _jax.jacrev(self._forward_fn)

    def _log_val(self, x, out):
        out[:] = (
            _np.asarray(self._forward_fn(self._params, x))
            .view(dtype=self._complex_dtype)
            .squeeze()
        )

    def _der_log(self, x, out):
        # Computes the Jacobian matrix
        J = self._jacobian(self._params, x)
        batch_size = x.shape[0]
        i = 0
        for g in (g.reshape(batch_size, 2, -1) for layer in J for g in layer):
            # NOTE: This can be avoided if one reorders parameters in other
            # places
            n = g.shape[2]
            out[:, i : i + n].real = g[:, 0, :]
            out[:, i : i + n].imag = g[:, 1, :]
            i += n

    def _is_holomorphic(self):
        return False

    def _set_parameters(self, new_parameters):
        state = []
        i = 0
        for layer in self._params:
            layer_state = ()
            for p in layer:
                if not _np.all(p.imag == 0):
                    raise ValueError("parameters are purely real")
                # NOTE: This relies on the fact that all our parameters are
                # stored in row major order
                layer_state += (
                    _np.ascontiguousarray(
                        new_parameters[i : i + p.size].real.reshape(p.shape)
                    ),
                )
                i += p.size
            state.append(layer_state)
        self._params = state

    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), _np.asarray(p).view()))
        return OrderedDict(state)

    def load_state_dict(self, other):
        state = []
        for i, layer in enumerate(self._params):
            layer_state = ()
            for j, p in enumerate(layer):
                name = str((i, j))
                if name not in other:
                    raise ValueError(
                        "state is missing required field {!r}".format(name)
                    )
                value = other[name]
                if p.shape != value.shape:
                    raise ValueError(
                        "field {!r} has wrong shape: {}; expected {}".format(
                            value.shape, p.shape
                        )
                    )
                layer_state += (value,)
            state.append(layer_state)
        self._params = state

    def save(self, filename):
        with open(filename, "wb") as output:
            dump(self.state_dict(), output)

    def load(self, filename):
        with open(filename, "rb") as input:
            self.load_state_dict(load(input))
