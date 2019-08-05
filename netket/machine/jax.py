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

from __future__ import absolute_import
from collections import OrderedDict
from functools import reduce
from pickle import dump, load
import os

import numpy as np

os.environ["JAX_ENABLE_X64"] = "1"
import jax

from .cxx_machine import CxxMachine

__all__ = ["Jax"]


class Jax(CxxMachine):
    def __init__(self, hilbert, module, seed=None):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet machine.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            seed: Seed to use to construct `jax.random.PRNGKey` for
                initialisation of parameters. If `None` seed will be chosen
                automatically (which is __not__ synchronized between MPI
                processes so prefer `self.init_random_parameters` if you're
                using multiple MPI tasks)
        """
        # NOTE: The following call to __init__ is important!
        super(Jax, self).__init__(hilbert)
        if seed is None:
            # NOTE(twesterhout): I believe, jax uses 32-bit integers internally
            # to represent the seed. Hence the limit
            seed = np.random.randint(2 ** 32 - 1)
        init_fn, self._forward_fn = module
        self._forward_fn = jax.jit(self._forward_fn)
        input_shape = (-1, hilbert.size)
        output_shape, self._params = init_fn(jax.random.PRNGKey(seed), input_shape)
        if output_shape != (-1, 2):
            raise ValueError("module's output_shape is weird, check your network")
        # Computes total number of parameters
        n_par = sum(reduce(lambda n, p: n + p.size, layer, 0) for layer in self._params)
        # Save the initial dtype (for debugging mostly)
        dtypes = set(p.dtype for layer in self._params for p in layer)
        if len(dtypes) != 1:
            raise ValueError("module parameters have different dtypes")
        self._dtype = next(iter(dtypes))
        self._complex_dtype = {
            np.dtype("float32"): np.complex64,
            np.dtype("float64"): np.complex128,
        }[self._dtype]
        assert all(
            np.asarray(p).flags.c_contiguous for layer in self._params for p in layer
        ), "sorry, column major order is not supported (yet)"
        self._n_par = lambda: n_par
        # Computes the Jacobian matrix using backprop
        self._jacobian = jax.jacrev(self._forward_fn)
        self._jacobian = jax.jit(self._jacobian)

    def _log_val(self, x, out):
        """
        Do not use this function directly! It's an implementation detail!

        See `self.log_val`.
        """
        out[:] = (
            np.asarray(self._forward_fn(self._params, x))
            .view(dtype=self._complex_dtype)
            .squeeze()
        )

    def _der_log(self, x, out):
        """
        Do not use this function directly! It's an implementation detail!

        See `self.log_val`.
        """
        J = self._jacobian(self._params, x)
        batch_size = x.shape[0]
        i = 0
        for g in (g.reshape(batch_size, 2, -1) for layer in J for g in layer):
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
                if not np.all(p.imag == 0):
                    raise ValueError("parameters are purely real")
                # NOTE: This relies on the fact that all our parameters are
                # stored in row major order
                layer_state += (
                    np.ascontiguousarray(
                        new_parameters[i : i + p.size].real.reshape(p.shape)
                    ),
                )
                i += p.size
            state.append(layer_state)
        self._params = state

    @property
    def dtype(self):
        """
        Datatype of the parameters.
        """
        return self._dtype

    def state_dict(self):
        state = []
        for i, layer in enumerate(self._params):
            for j, p in enumerate(layer):
                state.append((str((i, j)), np.asarray(p).view()))
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
