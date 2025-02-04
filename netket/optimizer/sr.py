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
import functools
from typing import Any
import warnings

import jax

from netket.vqs import VariationalState
from netket.utils.types import Scalar, ScalarOrSchedule
from netket.utils import struct

from .qgt import QGTAuto
from .preconditioner import AbstractLinearPreconditioner


def check_conflicting_args_in_partial(
    qgt: functools.partial | Any,
    conflicting_args: list[str],
    warning_message: str,
) -> None:
    """
    Check for conflicting arguments in a QGT partial.

    Args:
        qgt: The partial to perform the check or any other object.
        conflicting_args: List of argument names to check for conflicts.
        warning_message: Warning message for the user.

    Raises:
        UserWarning: If conflicting arguments are found.
    """

    if not isinstance(qgt, functools.partial):
        return

    specified_args = set(qgt.keywords.keys())
    conflicting = set(conflicting_args).intersection(specified_args)
    if conflicting:
        warnings.warn(
            warning_message.format(
                conflicting, {k: qgt.keywords[k] for k in conflicting}
            ),
            UserWarning,
            stacklevel=3,
        )


class SR(AbstractLinearPreconditioner, mutable=True):
    r"""
    Stochastic Reconfiguration or Natural Gradient preconditioner for the gradient.

    Constructs the structure holding the parameters for using the
    Stochastic Reconfiguration/Natural gradient method.

    This preconditioner changes the gradient :math:`\nabla_i E` such that the
    preconditioned gradient :math:`\Delta_j` solves the system of equations

    .. math::

        (S_{i,j} + \delta_{i,j}(\epsilon_1 S_{i,i} + \epsilon_2)) \Delta_{j} = \nabla_i E

    Where :math:`S` is the Quantum Geometric Tensor (or Fisher Information Matrix),
    preconditioned according to the diagonal scale :math:`\epsilon_1` (`diag_scale`)
    and the diagonal shift :math:`\epsilon_2` (`diag_shift`). The default
    regularisation takes :math:`\epsilon_1=0` and :math:`\epsilon_2=0.01`.

    Depending on the arguments, an implementation is chosen. For
    details on all possible kwargs check the specific SR implementations
    in the documentation.

    You can also construct one of those structures directly.

    .. warning::

        NetKet also has an experimental implementation of the SR preconditioner using
        the kernel trick, also known as MinSR. This implementation relies on inverting
        the :math:`T = X^T X` matrix, where :math:`X` is the Jacobian of wavefunction and
        is therefore much more efficient than the standard SR for very large numbers
        of parameters.

        Look at :class:`netket.experimental.driver.VMC_SRt` for more details.

    """

    diag_shift: ScalarOrSchedule = struct.field(serialize=False, default=0.01)
    """Diagonal shift added to the S matrix. Can be a Scalar value, an
       `optax <https://optax.readthedocs.io>`_ schedule or a Callable function."""

    diag_scale: ScalarOrSchedule | None = struct.field(serialize=False, default=None)
    """Diagonal shift added to the S matrix. Can be a Scalar value, an
       `optax <https://optax.readthedocs.io>`_ schedule or a Callable function."""

    qgt_constructor: Callable = struct.static_field(default=None)
    """The Quantum Geometric Tensor type or a constructor."""

    qgt_kwargs: dict = struct.field(serialize=False, default=None)
    """The keyword arguments to be passed to the Geometric Tensor constructor."""

    def __init__(
        self,
        qgt: Callable | None = None,
        solver: Callable = jax.scipy.sparse.linalg.cg,
        *,
        diag_shift: ScalarOrSchedule = 0.01,
        diag_scale: ScalarOrSchedule | None = None,
        solver_restart: bool = False,
        **kwargs,
    ):
        r"""
        Constructs the structure holding the parameters for using the
        Stochastic Reconfiguration/Natural gradient method.

        Depending on the arguments, an implementation is chosen. For
        details on all possible kwargs check the specific SR implementations
        in the documentation.

        You can also construct one of those structures directly.

        Args:
            qgt: The Quantum Geometric Tensor type to use.
            solver: (Defaults to :func:`jax.scipy.sparse.linalg.cg`) The method
                used to solve the linear system. Must be a jax-jittable
                function taking as input a pytree and outputting
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
            holomorphic: boolean indicating if the ansatz is holomorphic or not. May
                speed up computations for models with complex-valued parameters.
        """
        if qgt is None:
            qgt = QGTAuto(solver)

        self.qgt_constructor = qgt
        self.qgt_kwargs = kwargs
        self.diag_shift = diag_shift
        self.diag_scale = diag_scale

        check_conflicting_args_in_partial(
            qgt,
            ["diag_shift", "diag_scale"],
            "Constructing the SR object with `SR(qgt= MyQGTType({}))` can lead to unexpected results and has been deprecated, "
            "because the keyword arguments specified in the QGTType are overwritten by those specified by the SR class and its defaults.\n\n"
            "To fix this, construct SR as  `SR(qgt=MyQGTType, {})` .\n\n"
            "In the future, this warning will become an error.",
        )

        super().__init__(solver, solver_restart=solver_restart)

    def lhs_constructor(self, vstate: VariationalState, step: Scalar | None = None):
        """
        This method constructs the left-hand side (LHS) operator for the linear system.
        """
        diag_shift = self.diag_shift
        if callable(self.diag_shift):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_shift`, you must call "
                    "the preconditioner with an extra argument `step`."
                )
            diag_shift = diag_shift(step)

        diag_scale = self.diag_scale
        if callable(self.diag_scale):
            if step is None:
                raise TypeError(
                    "If you use a scheduled `diag_scale`, you must call "
                    "the preconditioner with an extra argument `step`."
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
