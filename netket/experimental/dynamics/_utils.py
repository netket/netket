# Copyright 2021 The NetKet Authors - All Rights Reserved.
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

from functools import wraps

import jax
import jax.numpy as jnp

import netket as nk
from netket import config
from netket.utils.types import Array, PyTree


LimitsType = tuple[float | None, float | None]
"""Type of the dt limits field, having independently optional upper and lower bounds."""

def expand_dim(tree: PyTree, sz: int):
    """
    creates a new pytree with same structure as input `tree`, but where very leaf
    has an extra dimension at 0 with size `sz`.
    """

    def _expand(x):
        return jnp.zeros((sz, *x.shape), dtype=x.dtype)

    return jax.tree_util.tree_map(_expand, tree)

def propose_time_step(
    dt: float, scaled_error: float, error_order: int, limits: LimitsType
):
    """
    Propose an updated dt based on the scheme suggested in Numerical Recipes, 3rd ed.
    """
    SAFETY_FACTOR = 0.95
    err_exponent = -1.0 / (1 + error_order)
    return jnp.clip(
        dt * SAFETY_FACTOR * scaled_error**err_exponent,
        limits[0],
        limits[1],
    )


def maybe_jax_jit(fun, *jit_args, **jit_kwargs):
    """
    Only jit if `config.netket_experimental_disable_ode_jit` is False.

    This is used to disable jitting when this config is set. The switch is
    performed at runtime so that the flag can be changed as desired.
    """

    # jit the function only once:
    jitted_fun = jax.jit(fun, *jit_args, **jit_kwargs)

    @wraps(fun)
    def _maybe_jitted_fun(*args, **kwargs):
        if config.netket_experimental_disable_ode_jit:
            with jax.spmd_mode("allow_all"):
                res = fun(*args, **kwargs)
            return res
        else:
            return jitted_fun(*args, **kwargs)

    return _maybe_jitted_fun


def set_flag_jax(condition, flags, flag):
    """
    If `condition` is true, `flags` is updated by setting `flag` to 1.
    This is equivalent to the following code, but compatible with jax.jit:
        if condition:
            flags |= flag
    """
    return jax.lax.cond(
        condition,
        lambda x: x | flag,
        lambda x: x,
        flags,
    )


def scaled_error(y, y_err, atol, rtol, *, last_norm_y=None, norm_fn):
    norm_y = norm_fn(y)
    scale = (atol + jnp.maximum(norm_y, last_norm_y) * rtol) / nk.jax.tree_size(y_err)
    return norm_fn(y_err) / scale, norm_y


def euclidean_norm(x: PyTree | Array):
    """
    Computes the Euclidean L2 norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.sqrt(jnp.sum(jnp.abs(x) ** 2))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + y,
                jax.tree_util.tree_map(lambda x: jnp.sum(jnp.abs(x) ** 2), x),
            )
        )


def maximum_norm(x: PyTree | Array):
    """
    Computes the maximum norm of the Array or PyTree intended as a flattened array
    """
    if isinstance(x, jnp.ndarray):
        return jnp.max(jnp.abs(x))
    else:
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                jnp.maximum,
                jax.tree_util.tree_map(lambda x: jnp.max(jnp.abs(x)), x),
            )
        )


def append_docstring(doc):
    """
    Decorator that appends the string `doc` to the decorated function.

    This is needed here because docstrings cannot be f-strings or manipulated strings.
    """

    def _append_docstring(fun):
        fun.__doc__ = fun.__doc__ + doc
        return fun

    return _append_docstring


args_fixed_dt_docstring = """
    Args:
        dt: Timestep (floating-point number).
"""

args_adaptive_docstring = """
    This solver is adaptive, meaning that the time-step is changed at every
    iteration in order to keep the error below a certain threshold.

    In particular, given the variables at step :math:`t`, :math:`\\theta^{t}` and the
    error at the same time-step, :math:`\\epsilon^t`, we compute a rescaled error by
    using the absolute (**atol**) and relative (**reltol**) tolerances according
    to this formula.

    .. math::

        \\epsilon^\\text{scaled} = \\text{Norm}(\\frac{\\epsilon^{t}}{\\epsilon_{atol} +
            \\max(\\theta^t, \\theta^{t-1})\\epsilon_{reltol}}),

    where :math:`\\text{Norm}` is a function that normalises the vector, usually a vector
    norm but could be something else as well, and :math:`\\max` is an elementwise maximum
    function (with lexicographical ordering for complex numbers).

    Then, the integrator will attempt to keep `\\epsilon^\\text{scaled}<1`.

    Args:
        dt: Timestep (floating-point number). When :code:`adaptive==False` this value
            is never changed, when :code:`adaptive == True` this is the initial timestep.
        adaptive: Whether to use adaptive timestepping (Defaults to False).
            Not all integrators support adaptive timestepping.
        atol: Maximum absolute error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. A value of 0 means it is ignored. Note that the `norm` used
            to compute the error can be  changed in the :ref:`netket.experimental.TDVP`
            driver. (Defaults to 0).
        rtol: Maximum relative error at every time-step during adaptive timestepping.
            A larger value will lead to larger timestep. This option is ignored if
            `adaptive=False`. Note that the `norm` used to compute the error can be
            changed in the :ref:`netket.experimental.TDVP` driver. (Defaults to 1e-7)
        dt_limits: A length-2 tuple of minimum and maximum timesteps considered by
            adaptive time-stepping. A value of None signals that there is no bound.
            Defaults to :code:`(None, 10*dt)`.
    """
