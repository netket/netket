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

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from netket import jax as nkjax
from netket.utils.types import Array, PyTree


@partial(jax.jit, static_argnames=("apply_fun"))
def is_probably_holomorphic(
    apply_fun: Callable[[PyTree, Array], Array],
    parameters: PyTree,
    samples: Array,
    model_state: PyTree | None = None,
) -> bool:
    r"""
    Check if a function :math:`\psi` is likely to be holomorphic almost
    everywhere.

    The check is done by verifying if the function satisfies on the provided
    samples the
    `Cauchy-Riemann equations <https://en.wikipedia.org/wiki/Cauchy–Riemann_equations>`_.
    In particular, assuming that the function is complex-valued
    :math:`\psi(\theta, x) = \psi_r(\theta, x) + i \psi_i(\theta, x)`
    where :math:`\psi_r` and :math:`\psi_i` are it's real and imaginary part,
    and the parameters can also be split in real and imaginary part according to
    :math:`\theta = \theta_r + i\theta_i`, the conditions real

    .. math::

        \frac{\partial \psi_r(\theta, x)}{\partial \theta_r} =
        \frac{\partial \psi_i(\theta, x)}{\partial \theta_i}
        \,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,
        \frac{\partial \psi_r(\theta, x)}{\partial \theta_i} =
        -\frac{\partial \psi_i(\theta, x)}{\partial \theta_r}

    Those conditions are always not verified if the parameters are
    real (:math:`\theta_i=0`), in which case we automatically return False.

    We verify the Cauchy-Riemann equations on the provided samples, so you should
    provide a sufficiently large number of points to be relatively sure if your
    function is holomorphic, because there are some functions that are locally non
    holomorphic, however that is rare and in general you can trust the result of this
    function.

    .. Note::

        If you are working with NetKet, you are most likely pretty good at math. In that
        case, instead of relying on the computer to tell you if the function is holomorphic
        or not you can do it by yourself! It's quite easy: the Cauchy-Riemann conditions
        can be rewritten using `Wirtinger derivatives <https://en.wikipedia.org/wiki/Wirtinger_derivatives>`_
        (where we treat the complex conjugate variables :math:`\theta` and :math:`\theta^\star`
        as linearly independent) as

        .. math::

            \frac{\partial \psi(\theta, x)]}{\partial \theta^\star} = 0 .

        Practically, this can be read as *if the function :math:`\psi` depends on
        :math:`\theta^\star` it will not be holomorphic.* Therefore, a very
        easy way to spot non-holomorphic functions is to verify if any operation
        among the following appear in the function:

        * complex conjugation
        * absolute value
        * real or imaginary part


    .. Note::

        In most Machine-Learning applications we are only interested in holomorphicity
        almost-everywhere, therefore if the function is not holomorphic on one particular
        point, such as :math:`\theta=0`, :math:`\theta=1` or similar this will in general
        not matter because that is a set of dimension zero that we will never cross.

        This is similar to the way the derivative of locally non-differentiable functions like
        the modulus is obtained: we assume that we never enter those regions.

    Args:
        apply_fun: The forward pass of the variational function. This should be a Callable
            accepting two inputs, the first being the parameters as a dictionary with at least
            a key :code:`params`, and possibly other keys given by the :code:`model_state`. The
            second argument to :code:`apply_fun` will be the samples.
        parameters : a pytree of parameters to be passed as the first argument to the :code:`apply_fun`.
            We will check if the ansatz is holomoprhic with respect to the derivatives of those
            parameters.
        samples : a set of samples.
        model_state: optional dictionary of state parameters of the model.

    """

    if nkjax.tree_leaf_isreal(parameters):
        return False

    samples = samples.reshape(-1, samples.shape[-1])
    jacs = nkjax.jacobian(
        apply_fun, parameters, samples, model_state=model_state, mode="complex"
    )

    # ∂ᵣψᵣ
    dr_dpr = jax.tree_util.tree_map(lambda x: x[:, 0, ...], jacs.real)
    # ∂ᵣψᵢ
    dr_dpi = jax.tree_util.tree_map(lambda x: x[:, 1, ...], jacs.real)
    # ∂ᵢψᵣ
    di_dpr = jax.tree_util.tree_map(lambda x: x[:, 0, ...], jacs.imag)
    # ∂ᵢψᵢ
    di_dpi = jax.tree_util.tree_map(lambda x: x[:, 1, ...], jacs.imag)

    # verify that ∂ᵣψᵣ == ∂ᵢψᵢ
    cond1 = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), dr_dpr, di_dpi)
    # verify that ∂ᵣψᵢ == -∂ᵢψᵣ
    cond2 = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, -y), dr_dpi, di_dpr)

    return jax.tree_util.tree_reduce(jnp.bitwise_and, (cond1, cond2))
