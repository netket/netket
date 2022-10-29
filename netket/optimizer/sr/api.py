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

from typing import Callable, Optional

from functools import partial
from collections import namedtuple
from dataclasses import dataclass

from netket.vqs import VariationalState
from netket.utils.numbers import is_scalar
from netket.utils.types import Scalar, ScalarOrSchedule

import jax

from ..qgt import QGTAuto
from ..preconditioner import AbstractLinearPreconditioner

Preconditioner = namedtuple("Preconditioner", ["object", "solver"])

default_iterative = "cg"
# default_direct = "eigen"


def build_SR(*args, solver_restart: bool = False, **kwargs):
    r"""
    Constructs the structure holding the parameters for using the
    Stochastic Reconfiguration/Natural gradient method.

    This preconditioner changes the gradient :math:`\nabla_i E` such that the
    preconditioned gradient :math:`\delta_j` solves the system of equations

    .. math::

        (S_{i,j} + \epsilon_1 S_{i,i} + \epsilon_2) \delta_{j} = \nabla_i E

    Where :math:`S` is the Quantum Geometric Tensor (or Fisher Information Matrix),
    preconditioned according to the diagonal scale :math:`\epsilon_1` (`diag_scale`)
    and the diagonal shift :math:`epsilon_2` (`diag_shift`). The default
    regularisation takes :math:`\epsilon_1=0` and :math:`\epsilon_2=0.01`.

    Depending on the arguments, an implementation is chosen. For
    details on all possible kwargs check the specific SR implementations
    in the documentation.

    You can also construct one of those structures directly.

    Args:
        qgt: The Quantum Geometric Tensor type to use.
        solver: The method used to solve the linear system. Must be a jax-
            jittable function taking as input a pytree and outputting
            a tuple of the solution and extra data.
        diag_shift: (Default `0.01`) Diagonal shift added to the S matrix. Can be
            a Scalar value, an `optax <https://optax.readthedocs.io>`_ schedule
            or a Callable function.
        diag_scale: (Default `0`) Scale of the shift proportional to the
            diagonal of the S matrix added added to it. Can be a Scalar value,
            an `optax <https://optax.readthedocs.io>`_ schedule or a
            Callable function.
        solver_restart: If False uses the last solution of the linear
            system as a starting point for the solution of the next
            (default=False).
        holomorphic: boolean indicating if the ansatz is boolean or not. May
            speed up computations for models with complex-valued parameters.
    """

    # try to understand if this is the old API or new
    # API

    old_api = False
    # new_api = False

    if "matrix" in kwargs:
        # new syntax
        return SR(*args, **kwargs)

    legacy_kwargs = ["iterative", "method"]
    legacy_solver_kwargs = ["tol", "atol", "maxiter", "M", "restart", "solve_method"]
    for key in legacy_kwargs + legacy_solver_kwargs:
        if key in kwargs:
            old_api = True
            break

    if len(args) > 0:
        if is_scalar(args[0]):  # it's diag_shift
            old_api = True
        # else:
        #    new_api = True

        if len(args) > 1:
            if isinstance(args[1], str):
                old_api = True
        #     else:
        #        new_api = True

    if old_api:
        for (i, arg) in enumerate(args):
            if i == 0:
                # diag shift
                kwargs["diag_shift"] = arg
            elif i == 1:
                kwargs["method"] = arg
            else:
                raise TypeError(
                    "SR takes at most 2 positional arguments but len(args) where provided"
                )

        args = tuple()

        solver = None
        if "iterative" in kwargs:
            kwargs.pop("iterative")
        if "method" in kwargs:
            legacy_solvers = {
                "cg": jax.scipy.sparse.linalg.cg,
                "gmres": jax.scipy.sparse.linalg.gmres,
            }
            if kwargs["method"] not in legacy_solvers:
                raise ValueError(
                    "The old API only supports cg and gmres solvers. "
                    "Migrate to the new API and use any solver from"
                    "jax.scipy.sparse.linalg"
                )
            solver = legacy_solvers[kwargs["method"]]
            kwargs.pop("method")
        else:
            solver = jax.scipy.sparse.linalg.cg

        solver_keys = {}
        has_solver_kw = False
        for key in legacy_solver_kwargs:
            if key in kwargs:
                solver_keys[key] = kwargs[key]
                has_solver_kw = True

        if has_solver_kw:
            solver = partial(solver, **solver_keys)

        kwargs["solver"] = solver

    return SR(*args, solver_restart=solver_restart, **kwargs)


@dataclass
class SR(AbstractLinearPreconditioner):
    r"""
    Stochastic Reconfiguration or Natural Gradient preconditioner for the gradient.

    This preconditioner changes the gradient :math:`\nabla_i E` such that the
    preconditioned gradient :math:`\delta_j` solves the system of equations

    .. math::

        (S_{i,j} + \epsilon_1 S_{i,i} + \epsilon_2) \delta_{j} = \nabla_i E

    Where :math:`S` is the Quantum Geometric Tensor (or Fisher Information Matrix),
    preconditioned according to the diagonal scale :math:`\epsilon_1` (`diag_scale`)
    and the diagonal shift :math:`epsilon_2` (`diag_shift`). The default
    regularisation takes :math:`\epsilon_1=0` and :math:`\epsilon_2=0.01`.
    """

    diag_shift: ScalarOrSchedule = 0.01
    """Diagonal shift added to the S matrix. Can be a Scalar value, an
       `optax <https://optax.readthedocs.io>`_ schedule or a Callable function."""

    diag_scale: Optional[ScalarOrSchedule] = None
    """Diagonal shift added to the S matrix. Can be a Scalar value, an
       `optax <https://optax.readthedocs.io>`_ schedule or a Callable function."""

    qgt_constructor: Callable = None
    """The Quantum Geometric Tensor type or a constructor."""

    qgt_kwargs: dict = None
    """The keyword arguments to be passed to the Geometric Tensor constructor."""

    def __init__(
        self,
        qgt: Optional[Callable] = None,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        *,
        diag_shift: ScalarOrSchedule = 0.01,
        diag_scale: Optional[ScalarOrSchedule] = None,
        solver_restart: bool = False,
        **kwargs,
    ):
        """
        Constructs the structure holding the parameters for using the
        Stochastic Reconfiguration/Natural gradient method.

        Depending on the arguments, an implementation is chosen. For
        details on all possible kwargs check the specific SR implementations
        in the documentation.

        You can also construct one of those structures directly.

        Args:
            qgt: The Quantum Geometric Tensor type to use.
            solver: The method used to solve the linear system. Must be a jax-
                jittable function taking as input a pytree and outputting
                a tuple of the solution and extra data.
            diag_shift: (Default `0.01`) Diagonal shift added to the S matrix. Can be
                a Scalar value, an `optax <https://optax.readthedocs.io>`_ schedule
                or a Callable function.
            diag_scale: (Default `0`) Scale of the shift proportional to the
                diagonal of the S matrix added added to it. Can be a Scalar value,
                an `optax <https://optax.readthedocs.io>`_ schedule or a
                Callable function.
            solver_restart: If False uses the last solution of the linear
                system as a starting point for the solution of the next
                (default=False).
            holomorphic: boolean indicating if the ansatz is boolean or not. May
                speed up computations for models with complex-valued parameters.
        """
        if qgt is None:
            qgt = QGTAuto(solver)

        self.qgt_constructor = qgt
        self.qgt_kwargs = kwargs
        self.diag_shift = diag_shift
        self.diag_scale = diag_scale
        super().__init__(solver, solver_restart=solver_restart)

    def lhs_constructor(self, vstate: VariationalState, step: Optional[Scalar] = None):
        """
        This method does things
        """
        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_shift`, you must call "
                    "the precoditioner with an extra argument `step`."
                )
            diag_shift = diag_shift(step)

        diag_scale = self.diag_scale
        if callable(self.diag_scale):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_scale`, you must call "
                    "the precoditioner with an extra argument `step`."
                )
            diag_scale = diag_scale(step)

        return self.qgt_constructor(
            vstate,
            diag_shift=diag_shift,
            diag_scale=diag_scale,
            **self.qgt_kwargs,
        )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            + f"\n  qgt_constructor = {self.qgt_constructor}, "
            + f"\n  diag_shift      = {self.diag_shift}, "
            + f"\n  diag_scale      = {self.diag_scale}, "
            + f"\n  qgt_kwargs      = {self.qgt_kwargs}, "
            + f"\n  solver          = {self.solver}, "
            + f"\n  solver_restart  = {self.solver_restart}"
            + ")"
        )
