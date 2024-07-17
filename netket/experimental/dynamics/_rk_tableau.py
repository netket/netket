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

from typing import Callable, Optional

import jax
import jax.numpy as jnp

from netket.utils.struct import dataclass
from netket.utils.types import Array, PyTree

default_dtype = jnp.float64


def expand_dim(tree: PyTree, sz: int):
    """
    creates a new pytree with same structure as input `tree`, but where very leaf
    has an extra dimension at 0 with size `sz`.
    """

    def _expand(x):
        return jnp.zeros((sz, *x.shape), dtype=x.dtype)

    return jax.tree_util.tree_map(_expand, tree)


@dataclass
class TableauRKExplicit:
    r"""
    Class representing the Butcher tableau of an explicit Runge-Kutta method [1,2],
    which, given the ODE dy/dt = F(t, y), updates the solution as

    .. math::
        y_{t+dt} = y_t + \sum_l b_l k_l

    with the intermediate slopes

    .. math::
        k_l = F(t + c_l dt, y_t + \sum_{m < l} a_{lm} k_m).

    If :code:`self.is_adaptive`, the tableau also contains the coefficients :math:`b'_l`
    which can be used to estimate the local truncation error by the formula

    .. math::
        y_{\mathrm{err}} = \sum_l (b_l - b'_l) k_l.

    [1] https://en.wikipedia.org/w/index.php?title=Runge%E2%80%93Kutta_methods&oldid=1055669759
    [2] J. Stoer and R. Bulirsch, Introduction to Numerical Analysis, Springer NY (2002).
    """

    order: tuple[int, int]
    """The order of the tableau"""
    a: jax.numpy.ndarray
    b: jax.numpy.ndarray
    c: jax.numpy.ndarray
    c_error: Optional[jax.numpy.ndarray]
    """Coefficients for error estimation."""

    @property
    def is_explicit(self):
        jnp.allclose(self.a, jnp.tril(self.a))  # check if lower triangular

    @property
    def is_adaptive(self):
        return self.b.ndim == 2

    @property
    def is_fsal(self):
        """Returns True if the first iteration is the same as last."""
        # TODO: this is not yet supported
        return False

    @property
    def stages(self):
        """
        Number of stages (equal to the number of evaluations of the ode function)
        of the RK scheme.
        """
        return len(self.c)

    @property
    def error_order(self):
        """
        Returns the order of the embedded error estimate for a tableau
        supporting adaptive step size. Otherwise, None is returned.
        """
        if not self.is_adaptive:
            return None
        else:
            return self.order[1]

    def _compute_slopes(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Computes the intermediate slopes k_l."""
        times = t + self.c * dt

        # TODO: Use FSAL

        k = expand_dim(y_t, self.stages)
        for l in range(self.stages):
            dy_l = jax.tree_util.tree_map(
                lambda k: jnp.tensordot(
                    jnp.asarray(self.a[l], dtype=k.dtype), k, axes=1
                ),
                k,
            )
            y_l = jax.tree_util.tree_map(
                lambda y_t, dy_l: jnp.asarray(y_t + dt * dy_l, dtype=dy_l.dtype),
                y_t,
                dy_l,
            )
            k_l = f(times[l], y_l, stage=l)
            k = jax.tree_util.tree_map(lambda k, k_l: k.at[l].set(k_l), k, k_l)

        return k

    def step(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """Perform one fixed-size RK step from `t` to `t + dt`."""
        k = self._compute_slopes(f, t, dt, y_t)

        b = self.b[0] if self.b.ndim == 2 else self.b
        y_tp1 = jax.tree_util.tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(b, dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )

        return y_tp1

    def step_with_error(
        self,
        f: Callable,
        t: float,
        dt: float,
        y_t: Array,
    ):
        """
        Perform one fixed-size RK step from `t` to `t + dt` and additionally return the
        error vector provided by the adaptive solver.
        """
        if not self.is_adaptive:
            raise RuntimeError(f"{self} is not adaptive")

        k = self._compute_slopes(f, t, dt, y_t)

        y_tp1 = jax.tree_util.tree_map(
            lambda y_t, k: y_t
            + jnp.asarray(dt, dtype=y_t.dtype)
            * jnp.tensordot(jnp.asarray(self.b[0], dtype=k.dtype), k, axes=1),
            y_t,
            k,
        )
        db = self.b[0] - self.b[1]
        y_err = jax.tree_util.tree_map(
            lambda k: jnp.asarray(dt, dtype=k.dtype)
            * jnp.tensordot(jnp.asarray(db, dtype=k.dtype), k, axes=1),
            k,
        )

        return y_tp1, y_err


@dataclass
class NamedTableau:
    name: str
    data: TableauRKExplicit

    def __repr__(self) -> str:
        return self.name
