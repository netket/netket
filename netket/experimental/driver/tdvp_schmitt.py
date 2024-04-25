# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
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

from typing import Callable, Union, Optional

from functools import partial

import jax
import jax.numpy as jnp

from netket import stats
from netket.operator import AbstractOperator
from netket.optimizer.qgt import QGTJacobianDense
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.vqs import VariationalState, VariationalMixedState, MCState
from netket.jax import tree_cast
from netket.utils import timing

from netket.experimental.dynamics import RKIntegratorConfig


from .tdvp_common import TDVPBaseDriver, odefun


class TDVPSchmitt(TDVPBaseDriver):
    r"""
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

    This driver, which only works with standard MCState variational states, uses the regularization
    procedure described in `M. Schmitt's PRL 125 <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.125.100503>`_ .

    With the force vector

    .. math::

        F_k=\langle \mathcal O_{\theta_k}^* E_{loc}^{\theta}\rangle_c

    and the quantum Fisher matrix

    .. math::

        S_{k,k'} = \langle \mathcal O_{\theta_k} (\mathcal O_{\theta_{k'}})^*\rangle_c

    and for real parameters :math:`\theta\in\mathbb R`, the TDVP equation reads

    .. math::

        q\big[S_{k,k'}\big]\theta_{k'} = -q\big[xF_k\big]

    Here, either :math:`q=\text{Re}` or :math:`q=\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\rho` can be included
    by increasing the diagonal entries and solving

    .. math::

        q\big[(1+\rho\delta_{k,k'})S_{k,k'}\big]\theta_{k'} = -q\big[F_k\big]

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

    .. math::

        S = V\Sigma V^\dagger

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed
    from the regularized inverted eigenvalues

    .. math::

        \tilde\sigma_k^{-1}=\frac{1}{\Big(1+\big(\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\big)^6\Big)\Big(1+\big(\frac{\epsilon_{SNR}}{\text{SNR}(\rho_k)}\big)^6\Big)}

    with :math:`\text{SNR}(\rho_k)` the signal-to-noise ratio of
    :math:`\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see
    `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).


    .. note::

        This TDVP Driver uses the time-integrators from the `nkx.dynamics` module, which are
        automatically executed under a `jax.jit` context.

        When running computations on GPU, this can lead to infinite hangs or extremely long
        compilation times. In those cases, you might try setting the configuration variable
        `nk.config.netket_experimental_disable_ode_jit = True` to mitigate those issues.

    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: RKIntegratorConfig,
        *,
        t0: float = 0.0,
        propagation_type: str = "real",
        holomorphic: Optional[bool] = None,
        diag_shift: float = 0.0,
        diag_scale: Optional[float] = None,
        error_norm: Union[str, Callable] = "qgt",
        rcond: float = 1e-14,
        rcond_smooth: float = 1e-8,
        snr_atol: float = 1,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
            holomorphic: a flag to indicate that the wavefunction is holomorphic.
            diag_shift: diagonal shift of the quantum geometric tensor (QGT)
            diag_scale: If not None rescales the diagonal shift of the QGT
            rcond : Cut-off ratio for small singular :math:`\sigma_k` values of the
                Quantum Geometric Tensor. For the purposes of rank determination,
                singular values are treated as zero if they are smaller than rcond times
                the largest singular value :code:`\sigma_{max}`.
            rcond_smooth : Smooth cut-off ratio for singular values of the Quantum Geometric
                Tensor. This regularization parameter used with a similar effect to `rcond`
                but with a softer curve. See :math:`\epsilon_{SVD}` in the formula
                above.
            snr_atol: Noise regularisation absolute tolerance, meaning that eigenvalues of
                the S matrix that have a signal to noise ratio above this quantity will be
                (soft) truncated. This is :math:`\epsilon_{SNR}` in the formulas above.

        """
        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.rcond = rcond
        self.rcond_smooth = rcond_smooth
        self.snr_atol = snr_atol

        self.diag_shift = diag_shift
        self.holomorphic = holomorphic
        self.diag_scale = diag_scale

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )


# Copyright notice:
# The function `_impl` below includes lines copied from the jVMC repository
# found at github.com/markusschmitt/vmc_jax and licensed according to
# MIT License, Copyright (c) 2021 Markus Schmitt


@timing.timed
@partial(jax.jit, static_argnames=("n_samples"))
def _impl(parameters, n_samples, E_loc, S, rhs_coeff, rcond, rcond_smooth, snr_atol):
    E = stats.statistics(E_loc)
    ΔE_loc = E_loc.reshape(-1, 1) - E.mean

    stack_jacobian = S.mode == "complex"

    O = S.O / jnp.sqrt(n_samples)  # already divided by jnp.sqrt(n_s)
    if stack_jacobian:
        O = O.reshape(-1, 2, S.O.shape[-1])
        O = O[:, 0, :] + 1j * O[:, 1, :]

    Sd = S.to_dense()
    ev, V = jnp.linalg.eigh(Sd)

    OEdata = O.conj() * ΔE_loc
    F = stats.sum(OEdata, axis=0)

    # Note: this implementation differs from Eq. 20 in Markus's paper, which I would
    # implement as `rho = mpi.mean(QEdata, axis=0)`. However, this is different from
    # changing the basis AFTER averaging over the samples, and leads to the wrong
    # normalisation of RHo.
    Q = jnp.tensordot(V.conj().T, O.T, axes=1).T
    QEdata = Q.conj() * ΔE_loc
    rho = V.conj().T @ F

    # Compute the SNR according to Eq. 21
    snr = jnp.abs(rho) * jnp.sqrt(n_samples) / jnp.sqrt(stats.var(QEdata, axis=0))

    # Discard eigenvalues below numerical precision
    ev_inv = jnp.where(jnp.abs(ev / ev[-1]) > rcond, 1.0 / ev, 0.0)
    # Set regularizer for singular value cutoff
    regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(ev / ev[-1])) ** 6)
    # Construct a soft cutoff based on the SNR
    regularizer2 = regularizer * (1.0 / (1.0 + (snr_atol / snr) ** 6))

    # solve the linear system by hand
    eta_p = ev_inv * regularizer2 * rhs_coeff * rho
    # convert back to the parameter space
    update = V @ eta_p

    # remainder of the solution
    rmd = jnp.linalg.norm(Sd.dot(update) - rhs_coeff * F) / jnp.linalg.norm(F)

    y, reassemble = convert_tree_to_dense_format(parameters, S.mode)
    update_tree = reassemble(update if jnp.iscomplexobj(y) else update.real)

    # If parameters are real, then take only real part of the gradient (if it's complex)
    dw = tree_cast(update_tree, parameters)

    return E, dw, rmd, snr


@odefun.dispatch
def odefun_schmitt(state: MCState, self: TDVPSchmitt, t, w, *, stage=0):  # noqa: F811
    # pylint: disable=protected-access

    state.parameters = w
    state.reset()

    op_t = self.generator(t)

    E_loc = state.local_estimators(op_t)

    self._S = QGTJacobianDense(
        state,
        diag_shift=self.diag_shift,
        diag_scale=self.diag_scale,
        holomorphic=self.holomorphic,
    )

    self._loss_stats, self._dw, self._rmd, self._snr = _impl(
        state.parameters,
        state.n_samples,
        E_loc,
        self._S,
        self._loss_grad_factor,
        self.rcond,
        self.rcond_smooth,
        self.snr_atol,
    )

    if stage == 0:  # TODO: This does not work with FSAL.
        self._last_qgt = self._S

    return self._dw


@partial(jax.jit, static_argnums=(3, 4))
def _map_parameters(forces, parameters, loss_grad_factor, propagation_type, state_T):
    forces = jax.tree_util.tree_map(
        lambda x, target: loss_grad_factor * x,
        forces,
        parameters,
    )

    forces = tree_cast(forces, parameters)

    return forces
