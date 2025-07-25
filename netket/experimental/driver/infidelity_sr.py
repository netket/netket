from typing import Any
from collections.abc import Callable

import jax
from jax.flatten_util import ravel_pytree
import flax

from netket import jax as nkjax
from netket.optimizer.solver import cholesky
from netket.vqs import MCState, FullSumState, VariationalState
from netket.utils import timing, struct
from netket.utils.types import ScalarOrSchedule, Optimizer, Array, PyTree
from netket.jax._jacobian.default_mode import JacobianMode
from netket.operator import AbstractOperator
from netket import stats as nkstats

from netket.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)
from netket._src.ngd.sr_srt_common import sr, srt
from netket._src.ngd.srt_onthefly import srt_onthefly
from netket._src.operator.hpsi_utils import make_logpsi_U_afun
from netket.experimental.observable.infidelity.expect import get_local_estimator

ApplyFun = Callable[[PyTree, Array], Array]
KernelArgs = tuple[ApplyFun, PyTree, Array, tuple[Any, ...]]
KernelFun = Callable[[PyTree, Array], KernelArgs]
DeriativesArgs = tuple[ApplyFun, PyTree, PyTree, Array]


@jax.jit
def _flatten_samples(x):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(x, 0, x.ndim - 1)


class Infidelity_SR(AbstractVariationalDriver):
    r"""
    Infidelity minimization with respect to a target state :math:`|\Phi\rangle` (with possibly an operator :math:`U` such that :math:`|\Phi\rangle \equiv U|\Phi\rangle`)
    using Variational Monte Carlo (VMC) and **Stochastic Reconfiguration/Natural Gradient Descent**.
    The optimization is analogous to the one of :class:`netket.experimental.driver.VMC_SR` for ground state.
    The infidelity :math:`I` among the variational state :math:`|\Psi\rangle` and the target state :math:`|\Phi\rangle` corresponds to:

    .. math::

        I = 1 - \frac{|\langle\Psi|\Phi\rangle|^2 }{ \langle\Psi|\Psi\rangle \langle\Phi|\Phi\rangle } = 1 - \frac{\langle\Psi|\hat{I}_{op}|\Psi\rangle }{ \langle\Psi|\Psi\rangle },

    where:

    .. math::

        \hat{I}_{op} = \frac{|\Phi\rangle\langle\Phi|}{\langle\Phi|\Phi\rangle}

    is the projector onto the target state :math:`|\Phi\rangle` which corresponds to an effective Hamiltonian.
    In this case, the effective local energy is :math:`H^{loc}(x) = \frac{\Phi(x)}{\Psi(x)} \mathbb{E}_{y \sim |\Phi(y)|^2}\left[\frac{\Psi(y)}{\Phi(y)}\right]`.

    For details see `Sinibaldi et al. <https://quantum-journal.org/papers/q-2023-10-10-1131/>`_ and `Gravina et al. <https://quantum-journal.org/papers/q-2025-07-22-1803/>`_.
    """

    target_state: VariationalState
    "The target variational state :math:`|\\Phi\rangle`."

    operator: AbstractOperator = None
    "Operator :math:`U`."

    cv_coeff: float = -0.5
    r"""
    Optional control variate coefficient for variance reduction in Monte Carlo estimation
    (see `Sinibaldi et al. <https://quantum-journal.org/papers/q-2023-10-10-1131/>`).
    If None, no control variate is used. Default to the optimal value -0.5.
    """

    # Settings set by user
    diag_shift: ScalarOrSchedule = struct.field(serialize=False)
    r"""
    The diagonal shift :math:`\lambda` in the curvature matrix.

    This can be a scalar or a schedule. If it is a schedule, it should be
    a function that takes the current step as input and returns the value of the shift.
    """
    proj_reg: ScalarOrSchedule = struct.field(serialize=False)

    momentum: bool = struct.field(serialize=False, default=False)
    r"""
    Flag specifying whether to use momentum in the optimisation.

    If `True`, the optimizer will use momentum to accumulate previous updates
    following the SPRING optimizer from
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    _mode: str = struct.field(serialize=False)
    _chunk_size_bwd: int | None = struct.field(serialize=False)
    _use_ntk: bool = struct.field(serialize=False)
    _on_the_fly: bool = struct.field(serialize=False)
    _linear_solver_fn: Any = struct.field(serialize=False)

    # Internal things cached
    _unravel_params_fn: Any = struct.field(serialize=False)

    # Serialized state
    _old_updates: PyTree = None
    _dp: PyTree = struct.field(serialize=False)
    info: Any | None = None
    """
    PyTree to pass on information from the solver,e.g, the quadratic model.
    """

    def __init__(
        self,
        target_state: VariationalState,
        optimizer: Optimizer,
        *,
        operator: AbstractOperator = None,
        diag_shift: ScalarOrSchedule,
        proj_reg: ScalarOrSchedule | None = None,
        momentum: ScalarOrSchedule | None = None,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        variational_state: MCState = None,
        chunk_size_bwd: int | None = None,
        mode: JacobianMode | None = None,
        use_ntk: bool | None = None,
        on_the_fly: bool | None = None,
    ):
        r"""
        Initialize the driver with the given arguments.

        .. warning::

            The optimizer should be an instance of `optax.sgd`. Other optimizers, while they might work,
            will not *make mathematical sense* in the context of the SR/NGD optimization.

        Args:
            target_state: The target state :math:`|\Phi\rangle` that must be matched.
            optimizer: The optimizer to use for the parameter updates. To perform proper
                SR/NGD optimization this should be an instance of `optax.sgd`, but can be
                any other optimizer if you are brave.
            operator: The operator :math:`U`.
            variational_state: The variational state to optimize.
            diag_shift: The diagonal regularization parameter :math:`\lambda` for the QGT/NTK.
            proj_reg: The regularization parameter for the projection of the updates.
                (This usually is not very important and can be left to None)
            momentum: (SPRING, disabled by default, read above for details) a number between [0,1]
                that specifies the damping factor of
                the previous updates and works somewhat similarly to the beta parameter of ADAM.
                The maximum amplification of  the step size in SPRING is
                :math:`A(\mu)=1/\sqrt{1-μ^2}`
                Thus the  amplification is at most a factor of :math:`A(0.9)=2.3` or
                :math:`A(0.99)=7.1`. Values around ``momentum = 0.8`` empirically work well.
                (Defaults to None)
            linear_solver_fn: The linear solver function to use for the NGD solver.
            mode: The mode used to compute the jacobian of the variational state.
                Can be `'real'` or `'complex'`. Real can be used for real-valued wavefunctions
                with a sign, to truncate the arbitrary phase of the wavefunction. This leads
                to lower computational cost.
            on_the_fly: Whether to compute the QGT or NTK using lazy evaluation methods.
                This usually requires less memory. (Defaults to None, which will
                automatically chose the potentially best method).
            chunk_size_bwd: The number of rows of the NTK or of the Jacobian evaluated
                in a single sweep.
            use_ntk: Wheter to compute the updates using the Neural Tangent Kernel (NTK)
                instead of the Quantum Geometric Tensor (QGT), aka switching between
                SR and minSR. (Defaults to None, which will automatically choose the best
                method)
        """

        if operator is not None:

            def _logpsi_fun(apply_fun, variables, x, *args):
                variables_applyfun, O = flax.core.pop(variables, "operator")

                xp, mels = O.get_conn_padded(x)
                xp = xp.reshape(-1, x.shape[-1])
                logpsi_xp = apply_fun(variables_applyfun, xp, *args)
                logpsi_xp = logpsi_xp.reshape(mels.shape)

                return jax.scipy.special.logsumexp(
                    logpsi_xp.astype(complex), axis=-1, b=mels
                )

            logUpsi_fun, new_variables = make_logpsi_U_afun(
                target_state._apply_fun, operator, target_state.variables
            )

            if isinstance(target_state, MCState):
                self.target_state = MCState(
                    sampler=target_state.sampler,
                    apply_fun=logUpsi_fun,
                    n_samples=target_state.n_samples,
                    variables=new_variables,
                )
                self.target_state.sampler_state = target_state.sampler_state

            if isinstance(target_state, FullSumState):
                self.target_state = FullSumState(
                    hilbert=target_state.hilbert,
                    apply_fun=logUpsi_fun,
                    variables=new_variables,
                )
        else:
            self.target_state = target_state

        if isinstance(variational_state, FullSumState):
            raise TypeError(
                "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
            )
        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        if use_ntk is None:
            use_ntk = variational_state.n_parameters > variational_state.n_samples
            print("Automatic SR implementation choice: ", "NTK" if use_ntk else "QGT")

        if on_the_fly is None:
            if use_ntk:
                on_the_fly = True
            else:
                on_the_fly = False
        elif on_the_fly and not use_ntk:
            raise ValueError(
                """
                `onthefly` is only supported when `use_ntk=True`.

                We plan to support this mode for the standard NGD in the future.
                In the meantime, use a standard VMC+SR QGTOnTheFly.
                """
            )

        self.diag_shift = diag_shift
        self.proj_reg = proj_reg
        self.momentum = momentum

        self.chunk_size_bwd = chunk_size_bwd
        self._use_ntk = use_ntk
        self.mode = mode
        self._on_the_fly = on_the_fly

        self._linear_solver_fn = linear_solver_fn

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)

        self._old_updates: PyTree = None
        self._dp: PyTree = None

        # PyTree to pass on information from the solver, e.g, the quadratic model
        self.info = None

        params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not nkjax.tree_ishomogeneous(params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex parameters. "
                "Hybrid structures are not yet supported (but we would welcome contributions. Get in touch with us!)"
            )

    @timing.timed
    def _forward_and_backward(self):
        self.state.reset()

        # Compute the local infidelity estimator and average Infidelity
        local_energies, local_energies_cv = get_local_estimator(
            self.state,
            self.target_state,
            self.cv_coeff,
        )
        self._loss_stats = nkstats.statistics(1 - local_energies_cv)

        # Extract the hyperparameters which might be iteration dependent
        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)

        if self.use_ntk:
            if self.on_the_fly:
                compute_sr_update_fun = srt_onthefly
            else:
                compute_sr_update_fun = srt
        else:
            if self.on_the_fly:
                raise NotImplementedError
            else:
                compute_sr_update_fun = sr

        samples = _flatten_samples(self.state.samples)
        self._dp, self._old_updates, self.info = compute_sr_update_fun(
            self.state._apply_fun,
            local_energies,
            self.state.parameters,
            self.state.model_state,
            samples,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver_fn,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd,
        )

        self._dp = jax.tree_util.tree_map(lambda x: -x, self._dp)

        return self._dp

    @timing.timed
    def _log_additional_data(self, log_dict: dict, step: int):
        """
        Method to be implemented in sub-classes of AbstractVariationalDriver to
        log additional data at every step.
        This method is called at every iteration when executing with `run`.

        Args:
            log_dict: The dictionary containing all logged data. It must be
                **modified in-place** adding new keys.
            step: the current step number.
        """
        # Always log the acceptance.
        if hasattr(self.state, "sampler_state"):
            acceptance = getattr(self.state.sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict["acceptance"] = acceptance

        # Log the quadratic model if requested.
        if self.info is not None:
            log_dict["info"] = self.info

    @property
    def mode(self) -> JacobianMode:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`, `'complex'`, or 'onthefly'.

        - `'real'` mode truncates imaginary part of the wavefunction, useful for real-valued wf with a sign.
        - `'complex'` is the general implementation that always works.
        - `onthefly` uses a lazy implementation of the neural tangent kernel and does not compute the jacobian.

        This internally uses :func:`netket.jax.jacobian`. See that function for a more complete documentation.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: str | JacobianMode | None):
        # TODO: Add support for 'onthefly' mode
        # At the moment, onthefly is only supported for use_ntk=True.
        # We raise a warning if the user tries to use it with use_ntk=False.
        if mode is None:
            mode = nkjax.jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.hilbert.random_state(jax.random.key(1), 3),
                warn=False,
            )

        # TODO: Add support for 'holomorphic' mode
        # At the moment we only support 'real' and 'complex' modes for jacobian.
        # We raise an error if the user tries to use 'holomorphic' mode.
        if mode not in ["complex", "real"]:
            raise ValueError(
                "`mode` only supports 'jacobian_real' for real-valued wavefunctions, and 'jacobian_complex' for complex valued wave functions."
                "`holomorphic` is not yet supported, but could be contributed in the future. \n"
                f"You gave {mode}"
            )

        self._mode = mode

    @property
    def on_the_fly(self) -> bool:
        """
        Whether
        """
        return self._on_the_fly

    @property
    def use_ntk(self) -> bool:
        r"""
        Whether to use the Neural Tangent Kernel (NTK) instead of the Quantum Geometric Tensor (QGT) to compute the update.
        """
        return self._use_ntk

    @property
    def chunk_size_bwd(self) -> int:
        """
        Chunk size for backward-mode differentiation. This reduces memory pressure at a potential cost of higher computation time.

        If computing the jacobian, the jacobian is computed in blocks of `chunk_size_bwd` rows.
        If computing the NTK lazily, this is the number of rows of NTK evaluated in a single sweep.
        The chunk size does not affect the result, up to numerical precision.
        """
        return self._chunk_size_bwd

    @chunk_size_bwd.setter
    def chunk_size_bwd(self, value: int | None):
        if not isinstance(value, int | None):
            raise TypeError("chunk_size must be an integer or None")
        self._chunk_size_bwd = value
