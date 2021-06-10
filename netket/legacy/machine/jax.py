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
from jax.experimental import stax

from collections import OrderedDict
import warnings
import igraph

from .abstract_machine import AbstractMachine

import numpy as _np
from jax import numpy as jnp
from jax import random
from netket.graph import Chain
from netket.legacy.random import randint as _randint
from netket import utils
from jax.tree_util import tree_flatten, tree_unflatten, tree_map, tree_leaves

from ._jax_utils import forward_apply, tree_size, grad as nk_grad, vjp as nk_vjp

# An extensible list of wrapping package modules like flax, haiku and others
jax_package_wrappers = []


def add_package_wrapper(condition_fun, wrapper_fun):
    """
    add_package_wrapper(condition_fun, wrapper_fun)

    Add support for a jax-based package.
    `condition_fun(module)` must return true if `module` is a module of this package,
    and `wrapper_fun(module)` must return the tuple of init and apply functions for
    this module.
    """
    jax_package_wrappers.append((condition_fun, wrapper_fun))


# Maybe wrap the module of some package.
def maybe_wrap_module(module):
    for (condition_fun, wrap_fun) in jax_package_wrappers:
        if condition_fun(module):
            return wrap_fun(module)

    return module


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
        super().__init__(hilbert=hilbert, dtype=dtype)

        self._init_fn, self._forward_fn_nj = maybe_wrap_module(module)

        self._forward_fn = lambda pars, x: forward_apply(pars, self._forward_fn_nj, x)

        self.jax_init_parameters()

        self._npar = tree_size(self._params)

    def jax_init_parameters(self, seed=None):
        """
        Uses the init function of the jax networks to generate a set of parameters.
        Beware that usually jax does not correctly set the imaginary part of
        networks, so for complex networks it MUST be followed by a call to
        init_random_parameters, unless your network correctly initialised the
        imaginary part.
        """
        if seed is None:
            seed = _randint(0, 2 ** 32 - 2, size=())

        input_shape = (-1, self.input_size)
        output_shape, params = self._init_fn(jax.random.PRNGKey(seed), input_shape)

        self._params = self._cast(params)

        # TODO
        # We should also test that the output_shape is
        # (1,) or () for a vector input_shape (N,), which is what
        # is provided by the sampler.
        #  Either add this check of fix the samplers
        if output_shape != (-1, 1) and output_shape != (-1,):
            raise ValueError(
                r"""
                A valid network must have 1 output, but the network
                you attempted to build, provided input_shape={} gave
                output_shape={}""".format(
                    input_shape, output_shape
                )
            )

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
            out = forward_apply(self._params, self._forward_fn_nj, x).reshape(
                x.shape[0],
            )
        else:
            out[:] = forward_apply(self._params, self._forward_fn_nj, x).reshape(
                x.shape[0],
            )
        return out

    @property
    def jax_forward(self):
        return self._forward_fn

    def der_log(self, x, out=None):
        if x.ndim != 2:
            raise RuntimeError("Invalid input shape, expected a 2d array")

        # Jax has bugs for R->C functions...
        out = nk_grad(self._params, self._forward_fn_nj, x)

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
            return nk_vjp(self._params, self._forward_fn_nj, x, vec, conjugate)

        else:

            if conjugate and self._dtype is complex:

                def prodj(j):
                    return jax.numpy.tensordot(vec.transpose(), j.conjugate(), axes=1)

            else:

                def prodj(j):
                    return jax.numpy.tensordot(vec.transpose().conjugate(), j, axes=1)

            jacobian = nk_grad(self._params, self._forward_fn_nj, x)
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
        assert tree_size(p) == self._npar
        self._params = p

    def numpy_flatten(self, data):
        r"""Returns a flattened numpy array representing the given data.
            This is typically used to serialize parameters and gradients.

        Args:
             data: a (possibly non-flat) structure containing jax arrays.

        Returns:
             numpy.ndarray: a one-dimensional array containing a copy of data
        """

        return _np.concatenate(tuple(fd.reshape(-1) for fd in tree_leaves(data)))

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
from jax.nn.initializers import glorot_normal, normal


def DenseReal(out_dim, use_bias=True, W_init=glorot_normal(), b_init=normal()):
    """Layer constructor function for a dense (fully-connected) layer with complex output but real weight."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = random.split(rng)
        if use_bias:
            Wr, Wi, br, bi = (
                W_init(k1, (input_shape[-1], out_dim)),
                W_init(k1, (input_shape[-1], out_dim)),
                b_init(k2, (out_dim,)),
                b_init(k2, (out_dim,)),
            )
            return output_shape, (Wr, Wi, br, bi)
        else:
            Wr, Wi = (
                W_init(k1, (input_shape[-1], out_dim)),
                W_init(k1, (input_shape[-1], out_dim)),
            )
            return output_shape, (Wr, Wi)

    def apply_fun(params, inputs, **kwargs):
        # distinguish between real and complex input because
        # jax is not smart enough to use the good gemm on his own.
        if jnp.iscomplexobj(inputs):
            if use_bias:
                Wr, Wi, br, bi = params
                return jnp.dot(inputs, Wr + 1j * Wi) + br + 1j * bi
            else:
                Wr, Wi = params
                return jnp.dot(inputs, Wr + 1j * Wi)
        else:
            if use_bias:
                Wr, Wi, br, bi = params
                return jnp.dot(inputs, Wr) + 1j * jnp.dot(inputs, Wi) + br + 1j * bi
            else:
                Wr, Wi = params
                return jnp.dot(inputs, Wr) + 1j * jnp.dot(inputs, Wi)

    return init_fun, apply_fun


def SumLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return inputs.sum(axis=-1)

    return init_fun, apply_fun


SumLayer = SumLayer()


def logcosh(x):
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)


LogCoshLayer = stax.elementwise(logcosh)


def JaxRbm(hilbert, alpha, dtype=complex):
    return Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer),
        dtype=dtype,
    )


def MPSPeriodic(hilbert, graph, bond_dim, diag=False, symperiod=None, dtype=complex):
    r"""
    Constructs a periodic Matrix Product State (MPS) for a quantum state of discrete
    degrees of freedom, wrapped as Jax machine.  The MPS is defined as

    .. math:: \Psi(s_1,\dots s_N) = \Tr(A[s_1]\dots A[s_N]),

    for arbitrary local quantum numbers :math:`s_i`, where :math:`A[s_1]` is a matrix
    of dimension (bdim,bdim), depending on the value of the local quantum number :math:`s_i`.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            bond_dim (int): Virtual dimension of the MPS tensors.
            diag (bool): Whether or not to use diagonal matrices in the MPS tensors.
                default=False
            symperiod (int): Periodicity in the chain of MPS tensors. The chain of
                MPS tensors is constructed as a sequence of identical unit cells
                consisting of symperiod tensors. if None, symperiod equals the
                number of physical degrees of freedom.
                default=None
            dtype: complex or float, whether the variational parameters of the MPS
                are real or complex. default=complex

        returns:
            Jax machine of the Matrix Product state.
    """
    return Jax(
        hilbert,
        mpsPeriodic(hilbert, graph, bond_dim, diag, symperiod, dtype),
        dtype=dtype,
    )


def mpsPeriodic(hilbert, graph, bond_dim, diag=False, symperiod=None, dtype=complex):
    # default standard deviation equals 1e-2
    normal_init = jax.nn.initializers.normal()

    L = hilbert.size
    phys_dim = hilbert.local_size
    diag = diag
    dtype = dtype

    # determine transformation from local states to indices
    local_states = jax.numpy.array(hilbert.local_states)
    loc_vals_spacing = jax.numpy.roll(local_states, -1)[0:-1] - local_states[0:-1]
    if jax.numpy.max(loc_vals_spacing) == jax.numpy.min(loc_vals_spacing):
        loc_vals_spacing = loc_vals_spacing[0]
    else:
        raise AssertionError(
            "JaxMpsPeriodic can only be used with evenly spaced hilbert local values"
        )
    loc_vals_bias = jax.numpy.min(local_states)

    # check whether graph is periodic chain
    chain_graph = Chain(graph.n_edges).to_igraph()
    if not igraph.Graph(edges=graph.edges()).isomorphic(chain_graph):
        warnings.warn(
            "Warning: graph is not isomorphic to chain with periodic boundary conditions",
            UserWarning,
        )

    # determine shape of unit cell
    if symperiod == None:
        symperiod = L
    if L % symperiod == 0 and symperiod > 0:
        if diag:
            unit_cell_shape = (symperiod, phys_dim, bond_dim)
        else:
            unit_cell_shape = (symperiod, phys_dim, bond_dim, bond_dim)
    else:
        raise AssertionError(
            "The number of degrees of freedom of the Hilbert space needs to be a multiple of the period of the MPS"
        )

    # define diagonal tensors with correct unit cell shape
    if diag:
        iden_tensors = jax.numpy.ones((symperiod, phys_dim, bond_dim), dtype=dtype)
    else:
        iden_tensors = jax.numpy.repeat(
            jax.numpy.eye(bond_dim, dtype=dtype)[jax.numpy.newaxis, :, :],
            symperiod * phys_dim,
            axis=0,
        )
        iden_tensors = iden_tensors.reshape(symperiod, phys_dim, bond_dim, bond_dim)

    def init_fun(rng, input_shape):
        rng_re, rng_im = jax.random.split(rng)
        random_tensors_real = normal_init(rng_re, unit_cell_shape)
        if dtype == complex:
            random_tensors_imag = normal_init(rng_im, unit_cell_shape)
            random_tensors = random_tensors_real + 1j * random_tensors_imag
        else:
            random_tensors = random_tensors_real

        tensors = random_tensors + iden_tensors

        return (-1, 1), (tensors)

    def apply_fun(params, x, **kwargs):
        # expand diagonal to square matrices if diagonal mps
        if diag:
            params = jax.numpy.einsum(
                "ijk,kl->ijkl", params, jax.numpy.eye(params.shape[-1])
            )

        # create all tensors in mps from unit cell
        all_tensors = jax.numpy.tile(params, (L // symperiod, 1, 1, 1))

        # transform input to indices
        x = (x - loc_vals_bias) / loc_vals_spacing
        if len(x.shape) == 1:  # batch size is one
            x = jax.numpy.expand_dims(x, 0)

        def select_tensor(tensor, index):
            return tensor[index.astype(int)]

        def select_all_tensors(all_tensors, indices):
            return jax.vmap(select_tensor)(all_tensors, indices)

        # select right tensors using input for matrix multiplication
        selected_tensors = jax.vmap(select_all_tensors, (None, 0))(all_tensors, x)

        # create loop carry, in this case a unit matrix
        edges = jax.numpy.repeat(
            jax.numpy.eye(bond_dim, dtype=selected_tensors.dtype)[
                jax.numpy.newaxis, :, :
            ],
            selected_tensors.shape[0],
            axis=0,
        )

        def trace_mps(tensors, edge):
            def multiply_tensors(left_tensor, right_tensor):
                return jax.numpy.einsum("ij,jk->ik", left_tensor, right_tensor), None

            edge, _ = jax.lax.scan(multiply_tensors, edge, tensors)

            return jax.numpy.trace(edge)

        # trace the matrix multiplication
        z = jax.vmap(trace_mps)(selected_tensors, edges)
        return z

    return init_fun, apply_fun


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


def jaxRbmSpinPhase(hilbert, alpha):
    return stax.serial(
        stax.FanOut(2),
        stax.parallel(
            stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer),
            stax.serial(stax.Dense(alpha * hilbert.size), LogCoshLayer, SumLayer),
        ),
        FanInSum2ModPhase,
    )


def JaxRbmSpinPhase(hilbert, alpha, dtype=float):
    return Jax(
        hilbert,
        jaxRbmSpinPhase(hilbert, alpha),
        dtype=dtype,
    )
