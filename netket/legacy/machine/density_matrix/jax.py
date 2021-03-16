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
import jax.numpy as jnp
from jax import random

from .abstract_density_matrix import AbstractDensityMatrix
from ..jax import Jax as JaxPure, logcosh
from functools import partial


class Jax(JaxPure, AbstractDensityMatrix):
    def __init__(self, hilbert, module, dtype=complex):
        """
        Wraps a stax network (which is a tuple of `init_fn` and `predict_fn`)
        so that it can be used as a NetKet density matrix.

        Args:
            hilbert: Hilbert space on which the state is defined. Should be a
                subclass of `netket.hilbert.Hilbert`.
            module: A pair `(init_fn, predict_fn)`. See the documentation of
                `jax.experimental.stax` for more info.
            dtype: either complex or float, is the type used for the weights.
                In both cases the module must have a single output.
        """
        AbstractDensityMatrix.__init__(self, hilbert, dtype)
        JaxPure.__init__(self, hilbert, module, dtype)

        assert self.input_size == self.hilbert.size * 2

    @staticmethod
    @jax.jit
    def _dminput(xr, xc):
        if xc is None:
            x = xr
        else:
            x = jnp.hstack((xr, xc))
        return x

    def log_val(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return JaxPure.log_val(self, x, out=out)

    def der_log(self, xr, xc=None, out=None):
        x = self._dminput(xr, xc)

        return JaxPure.der_log(self, x, out=out)

    def diagonal(self):
        from .diagonal import Diagonal

        diag = Diagonal(self)

        def diag_jax_forward(params, x):
            return self.jax_forward(params, self._dminput(x, x))

        diag.jax_forward = diag_jax_forward

        return diag


from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.nn.initializers import glorot_normal, normal


def DensePurificationComplex(
    out_pure, out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (2 * out_pure + out_mix,)

        k = jax.random.split(rng, 7)

        input_size = input_shape[-1] // 2

        # Weights for the pure part
        Wr, Wi = (
            W_init(k[0], (input_size, out_pure)),
            W_init(k[1], (input_size, out_pure)),
        )

        # Weights for the mixing part
        Vr, Vi = (
            W_init(k[2], (input_size, out_mix)),
            W_init(k[3], (input_size, out_mix)),
        )

        if use_hidden_bias:
            br, bi = (b_init(k[4], (out_pure,)), b_init(k[5], (out_pure,)))
            cr = b_init(k[6], (out_mix,))

            return output_shape, (Wr, Wi, Vr, Vi, br, bi, cr)
        else:
            return output_shape, (Wr, Wi, Vr, Vi)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            Wr, Wi, Vr, Vi, br, bi, cr = params
        else:
            Wr, Wi, Vr, Vi = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        thetar = jax.numpy.dot(
            xr[
                :,
            ],
            (Wr + 1.0j * Wi),
        )
        thetac = jax.numpy.dot(
            xc[
                :,
            ],
            (Wr - 1.0j * Wi),
        )

        thetam = jax.numpy.dot(
            xr[
                :,
            ],
            (Vr + 1.0j * Vi),
        )
        thetam += jax.numpy.dot(
            xc[
                :,
            ],
            (Vr - 1.0j * Vi),
        )

        if use_hidden_bias:
            thetar += br + 1.0j * bi
            thetac += br - 1.0j * bi
            thetam += 2 * cr

        return jax.numpy.hstack((thetar, thetam, thetac))

    return init_fun, apply_fun


from ..jax import LogCoshLayer, SumLayer


def ndmSpin(hilbert, alpha, beta, use_hidden_bias=True):
    return stax.serial(
        DensePurificationComplex(
            int(alpha * hilbert.size), int(beta * hilbert.size), use_hidden_bias
        ),
        LogCoshLayer,
        SumLayer,
    )


def NdmSpin(hilbert, alpha, beta, use_hidden_bias=True):
    r"""
    A fully connected Neural Density Matrix (DBM). This type density matrix is
    obtained purifying a RBM with spin 1/2 hidden units.

    The number of purification hidden units can be chosen arbitrarily.

    The weights are taken to be complex-valued. A complete definition of this
    machine can be found in Eq. 2 of Hartmann, M. J. & Carleo, G.,
    Phys. Rev. Lett. 122, 250502 (2019).

    Args:
        hilbert: Hilbert space of the system.
        alpha: `alpha * hilbert.size` is the number of hidden spins used for
                the pure state part of the density-matrix.
        beta: `beta * hilbert.size` is the number of hidden spins used for the purification.
            beta=0 for example corresponds to a pure state.
        use_hidden_bias: If ``True`` bias on the hidden units is taken.
                         Default ``True``.
    """
    return Jax(
        hilbert,
        ndmSpin(hilbert, alpha, beta, use_hidden_bias),
        dtype=float,
    )


def DenseMixingReal(
    out_mix, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    """Layer constructor function for a complex purification layer."""

    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        output_shape = input_shape[:-1] + (out_mix,)

        k = jax.random.split(rng, 3)

        input_size = input_shape[-1] // 2

        # Weights for the mixing part
        Ur, Ui = (
            W_init(k[0], (input_size, out_mix)),
            W_init(k[1], (input_size, out_mix)),
        )

        if use_hidden_bias:
            dr = b_init(k[2], (out_mix,))

            return output_shape, (Ur, Ui, dr)
        else:
            return output_shape, (Ur, Ui)

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            Ur, Ui, dr = params
        else:
            Ur, Ui = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        theta = jax.numpy.dot(
            xr[
                :,
            ],
            (0.5 * Ur + 0.5j * Ui),
        )
        theta += jax.numpy.dot(
            xc[
                :,
            ],
            (0.5 * Ur - 0.5j * Ui),
        )

        if use_hidden_bias:
            theta += dr

        return theta

    return init_fun, apply_fun


def DensePureRowCol(
    out_pure, use_hidden_bias=True, W_init=glorot_normal(), b_init=normal()
):
    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        input_size = input_shape[-1] // 2

        single_output_shape = input_shape[:-1] + (out_pure,)
        output_shape = (single_output_shape, single_output_shape)

        k = jax.random.split(rng, 3)

        W = W_init(k[0], (input_size, out_pure))

        if use_hidden_bias:
            b = b_init(k[2], (out_pure,))

            return output_shape, (W, b)
        else:
            return output_shape, (W,)

    def apply_fun(params, inputs, **kwargs):
        if use_hidden_bias:
            W, b = params
        else:
            W = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        thetar = jax.numpy.dot(
            xr[
                :,
            ],
            W,
        )
        thetac = jax.numpy.dot(
            xc[
                :,
            ],
            W,
        )

        if use_hidden_bias:
            thetar += b
            thetac += b

        return (thetar, thetac)

    return init_fun, apply_fun


def FanInSum2():
    """Layer construction function for a fan-in sum layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0]
        return output_shape, tuple()

    def apply_fun(params, inputs, **kwargs):
        output = 0.5 * (inputs[0] + inputs[1])
        return output

    return init_fun, apply_fun


FanInSum2 = FanInSum2()


def FanInSub2():
    """Layer construction function for a fan-in sum layer."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[0]
        return output_shape, tuple()

    def apply_fun(params, inputs, **kwargs):
        output = 0.5j * (inputs[0] - inputs[1])
        return output

    return init_fun, apply_fun


FanInSub2 = FanInSub2()


def BiasRealModPhase(b_init=normal()):
    def init_fun(rng, input_shape):
        assert input_shape[-1] % 2 == 0
        input_size = input_shape[-1] // 2

        output_shape = input_shape[:-1]

        k = jax.random.split(rng, 2)

        br = b_init(k[0], (input_size,))
        bj = b_init(k[1], (input_size,))

        return output_shape, (br, bj)

    def apply_fun(params, inputs, **kwargs):
        br, bj = params

        xr, xc = jax.numpy.split(inputs, 2, axis=-1)

        biasr = jax.numpy.dot(
            (xr + xc)[
                :,
            ],
            br,
        )
        biasj = jax.numpy.dot(
            (xr - xc)[
                :,
            ],
            bj,
        )

        return 0.5 * biasr + 0.5j * biasj

    return init_fun, apply_fun


def ndmSpinPhase(hilbert, alpha, beta, use_hidden_bias=True, use_visible_bias=True):
    mod_pure = stax.serial(
        DensePureRowCol(alpha * hilbert.size, use_hidden_bias),
        stax.parallel(LogCoshLayer, LogCoshLayer),
        stax.parallel(SumLayer, SumLayer),
        FanInSum2,
    )

    phs_pure = stax.serial(
        DensePureRowCol(alpha * hilbert.size, use_hidden_bias),
        stax.parallel(LogCoshLayer, LogCoshLayer),
        stax.parallel(SumLayer, SumLayer),
        FanInSub2,
    )

    mixing = stax.serial(
        DenseMixingReal(int(beta * hilbert.size), use_hidden_bias),
        LogCoshLayer,
        SumLayer,
    )

    if use_visible_bias:
        biases = BiasRealModPhase()
        net = stax.serial(
            stax.FanOut(4),
            stax.parallel(mod_pure, phs_pure, mixing, biases),
            stax.FanInSum,
        )
    else:
        net = stax.serial(
            stax.FanOut(3),
            stax.parallel(mod_pure, phs_pure, mixing),
            stax.FanInSum,
        )

    return net


def NdmSpinPhase(hilbert, alpha, beta, use_hidden_bias=True, use_visible_bias=True):
    r"""
    A fully connected Neural Density Matrix (DBM). This type density matrix is
    obtained purifying a RBM with spin 1/2 hidden units.

    The number of purification hidden units can be chosen arbitrarily.

    The weights are taken to be complex-valued. A complete definition of this
    machine can be found in Eq. 2 of Hartmann, M. J. & Carleo, G.,
    Phys. Rev. Lett. 122, 250502 (2019).

    Args:
        hilbert: Hilbert space of the system.
        alpha: `alpha * hilbert.size` is the number of hidden spins used for
                the pure state part of the density-matrix.
        beta: `beta * hilbert.size` is the number of hidden spins used for the purification.
            beta=0 for example corresponds to a pure state.
        use_hidden_bias: If ``True`` bias on the hidden units is taken.
                         Default ``True``.
    """
    return Jax(
        hilbert,
        ndmSpinPhase(hilbert, alpha, beta, use_hidden_bias, use_visible_bias),
        dtype=float,
    )


def JaxRbmSpin(hilbert, alpha, dtype=complex):
    return Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * hilbert.size * 2), LogCoshLayer, SumLayer),
        dtype=dtype,
    )
