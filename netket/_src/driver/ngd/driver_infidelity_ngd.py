from typing import Callable, Optional, Union


from netket import jax as nkjax
from netket.optimizer.solver import cholesky
from netket.vqs import MCState, VariationalState
from netket.jax._jacobian.default_mode import JacobianMode
from netket.utils import timing, struct
from netket.utils.types import ScalarOrSchedule, Optimizer, Array
from netket.operator import DiscreteOperator, DiscreteJaxOperator

from netket_pro._src.operator.jax_utils import to_jax_operator

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)
from advanced_drivers._src.distribution.default import DefaultDistribution
from advanced_drivers._src.distribution.linear_operator import (
    LinearOperatorStateDistribution,
)
from advanced_drivers._src.distribution.constructors.base import (
    AbstractDistributionConstructor,
)
from advanced_drivers._src.driver.ngd.driver_abstract_ngd import (
    AbstractNGDDriver,
    DerivativesArgs,
    KernelFun,
    _get_samples_and_weights,
    _initialise_distribution,
)
from advanced_drivers._src.driver.ngd.infidelity_kernels import (
    cmc_kernel,
    smc_kernel,
)


class InfidelityOptimizerNG(AbstractNGDDriver):
    r"""
    Infidelity minimisation using Natural Gradient Descent between two states,
    with possibly two operators U and V acting on the states.

    .. math::

        \text{Loss}(\theta) = \langle \psi_\theta | V^{-1} U | \phi \rangle

    where :math:`\psi_\theta` is the variational state, :math:`\phi` is the target state,
    and :math:`U` and :math:`V` are the operators acting on the states.

    .. note::

        The algorithm implemented here comes from Luca's preprint
        "Neural Projected Quantum Dynamics: a systematic study" (https://arxiv.org/abs/2410.10720).

    .. warning::

        This driver does not support full summation calculations. To use full summation,
        use any driver in ``netket_pro``.

    """

    chunk_size_U: int | None = struct.field(pytree_node=False, serialize=False)
    """
    The chunk size to use for the forward pass (jacobian or vjp evaluation) of the target state.
    """
    chunk_size_V: int | None = struct.field(pytree_node=False, serialize=False)
    """
    The chunk size to use for the forward pass (jacobian or vjp evaluation) of the variational state.
    """

    _kernel_fun: Callable = struct.field(pytree_node=False, serialize=False)

    _target: VariationalState = struct.field(pytree_node=False, serialize=True)
    _cv_coeff: float | None = struct.field(serialize=False)
    _U_target: DiscreteJaxOperator | None = struct.field(
        pytree_node=False, serialize=False
    )
    _V_state: DiscreteJaxOperator | None = struct.field(
        pytree_node=False, serialize=False
    )
    _resample_fraction: float | None = struct.field(pytree_node=False, serialize=False)

    _importance_sampling_target_distribution: AbstractDistribution | None = (
        struct.field(pytree_node=False, serialize=False)
    )

    def __init__(
        self,
        target_state: MCState,
        optimizer: Optimizer,
        *,
        importance_sampling_distribution: AbstractDistribution | None = None,
        importance_sampling_target_distribution: AbstractDistribution | None = None,
        diag_shift: ScalarOrSchedule,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        collect_quadratic_model: bool = False,
        U: Optional[DiscreteOperator] = None,
        V: Optional[DiscreteOperator] = None,
        cv_coeff: Optional[float] = -0.5,
        resample_fraction: Optional[float] = None,
        estimator: Union[str, Callable] = "cmc",
        mode: Optional[JacobianMode] = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = None,
        sample_Uphi: bool | None = None,
    ):
        r"""
        Initialize the driver.

        Args:
            target_state: The target state (must be an :class:`nk.vqs.MCState`).
            optimizer: Determines how optimization steps are performed given the bare infidelity gradient.
                This should in general be an :func:`optax.sgd`.
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other variational states are not supported.
            diag_shift: The diagonal shift :math:`\lambda` of the curvature (S) matrix.
            proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
            momentum: Momentum used to accumulate updates in SPRING.
            linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
                Defaults to :func:`netket.optimizer.solver.cholesky`.
            mode: The mode used to compute the jacobian or vjp of the variational state.
                Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
                `real` can be used for real wavefunctions with a sign to further reduce the computational costs.
            on_the_fly: Whether to compute the QGT or NTK matrix without evaluating the full jacobian. Defaults to True.
                This ususally lowers the memory requirement and is necessary for large calculations.
            chunk_size_bwd: The chunk size to use for the backward pass (jacobian or vjp evaluation).
            collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are
                the linear and quadratic term in the approximation of the loss function. They are stored
                in the info dictionary of the driver.
            use_ntk: Whether to use the NTK for the computation of the updates.
            estimator: The estimator used to compute the local gradients and losses. Can be
                "cmc" or "smc". The good one is cmc (default).
            U: The operator :math:`\hat{U}` acting on the target state :math:`\phi`.
            V: The operator :math:`\hat{V}` whose inverse acts on the variational state :math:`\psi`.
            cv_coeff: The control variate coefficient. When minimising the infidelity, this should be set to -0.5.
            resample_fraction: The fraction of samples to resample at each step. Standard approaach is to
                keep this to None, which is equivalent to 1.0.
            sample_Uphi: Whether to sample the transformed target state :math:`\hat U | \phi \rangle` or
                directly or use importance sampling to sample from :math:`| \phi \rangle` (default=True, should not be
                specified if you are using importance sampling) WE MIGHT DEPRECATE THIS.
        """
        U_target = to_jax_operator(U) if U is not None else None
        V_state = to_jax_operator(V) if V is not None else None

        self.chunk_size_U = None
        if target_state.chunk_size is not None:
            # TODO: This means that the smallest chunk size is the max conn size of the operator U.
            # A better approach should be thought.
            self.chunk_size_U = max(
                target_state.chunk_size // getattr(U_target, "max_conn_size", 1), 1
            )

        self.chunk_size_V = None
        if variational_state.chunk_size is not None:
            # TODO: See note above.
            self.chunk_size_V = max(
                variational_state.chunk_size // getattr(V_state, "max_conn_size", 1), 1
            )

        self._target = target_state
        self._cv_coeff = cv_coeff
        self._U_target = U_target
        self._V_state = V_state

        if importance_sampling_target_distribution is None and sample_Uphi is None:
            sample_Uphi = True
        elif (
            importance_sampling_target_distribution is not None
            and sample_Uphi is not None
        ):
            raise ValueError(
                """
                You cannot specify both `importance_sampling_target_distribution` and `sample_Uphi`.
                If you want to use importance sampling, do not set `sample_Uphi` (or set it to None).

                The default is to sample the transformed target state (sample_Uphi=True).

                If you do not want to sample the transformed target state, we reccomend you use
                importance sampling directly by specifying

                    importance_sampling_target_distribution=advd.distributions.constructors.OverdispersedTargetStateDistribution(sample_transformed=False)

                Which will be equivalent to setting `sample_Uphi=False` but can also be combined with
                alpha=XYZ (default 2.0).
                """
            )
        if sample_Uphi is False:
            importance_sampling_target_distribution = DefaultDistribution()

        # Construct the importance sampling distribution specifications.
        if isinstance(
            importance_sampling_distribution, AbstractDistributionConstructor
        ):
            importance_sampling_distribution = (
                importance_sampling_distribution.construct_distribution(
                    driver=self,
                    which="state",
                )
            )
        if isinstance(
            importance_sampling_target_distribution, AbstractDistributionConstructor
        ):
            importance_sampling_target_distribution = (
                importance_sampling_target_distribution.construct_distribution(
                    driver=self,
                    which="target",
                )
            )

        self._importance_sampling_target_distribution = (
            importance_sampling_target_distribution
        )
        super().__init__(
            optimizer=optimizer,
            importance_sampling_distribution=importance_sampling_distribution,
            diag_shift=diag_shift,
            proj_reg=proj_reg,
            momentum=momentum,
            linear_solver_fn=linear_solver_fn,
            variational_state=variational_state,
            chunk_size_bwd=chunk_size_bwd,
            collect_quadratic_model=collect_quadratic_model,
            mode=mode,
            use_ntk=use_ntk,
            on_the_fly=on_the_fly,
            minimized_quantity_name="Infidelity",
        )

        self._resample_fraction = resample_fraction
        self._kernel = estimator

        # initialise the sampler
        _initialise_distribution(
            self.target,
            self.original_target_distribution,
            self.importance_sampling_target_distribution,
            chain_name=None,
        )

    def reset_step(self, hard: bool = False):
        """
        Resets the state of the driver at the beginning of a new step.

        This method is called at the beginning of every step in the optimization.
        """
        if hard:
            self.state.reset_hard()
            self.target.reset_hard()
        else:
            self.state.reset()
            self.target.reset()

    @timing.timed
    def _prepare_derivatives(self) -> DerivativesArgs:
        # Samples & weights from the variational state (psi)
        log_prob_p_fun, variables_p, samples, weights = _get_samples_and_weights(
            state=self.state,
            original_distribution=self.original_distribution,
            importance_sampling_distribution=self.importance_sampling_distribution,
            chunk_size=self.chunk_size_V,
            # chain_name="Vpsi" if self.V_state is not None else "default",
        )

        # Samples & weights from the target state (phi)
        log_prob_t_p_fun, variables_t_p, samples_t, weights_t = (
            _get_samples_and_weights(
                state=self.target,
                original_distribution=self.original_target_distribution,
                importance_sampling_distribution=self.importance_sampling_target_distribution,
                chunk_size=self.chunk_size_U,
                # chain_name="Uphi" if self.U_target is not None else "default",
            )
        )

        extra_args = (
            log_prob_t_p_fun,
            variables_t_p,
            samples_t,
            weights_t,
            self.cv_coeff,
        )

        # return log_prob, variables, samples, weights, extra_args
        return log_prob_p_fun, variables_p, samples, weights, extra_args

    @property
    def _kernel(self) -> KernelFun:
        """
        The kernel used to compute the local gradients and losses.
        The kernel function can be passed as a callable (don't forget to Jit).
        A predefined kernel can be selected by passing a string.

        Possible values for the predefined kernels are:
        - "cmc": the conditional Monte Carlo estimator (the best)
        - "smc": the simple Monte Carlo estimator (the intermediate)
        """
        return nkjax.HashablePartial(
            self._kernel_fun,
            chunk_size_U=self.chunk_size_U,
            chunk_size_V=self.chunk_size_V,
        )

    @_kernel.setter
    def _kernel(self, value: KernelFun | str):
        if type(value) is str:
            if value == "cmc":
                self._kernel_fun = cmc_kernel
            elif value == "smc":
                self._kernel_fun = smc_kernel
            else:
                raise ValueError(
                    f"Unknown estimator {value} (possible choices are 'smc' or 'cmc')"
                )
        elif callable(value):
            self._kernel_fun = value
        else:
            raise ValueError("The kernel must be a callable or a string")

    @timing.timed
    def _log_additional_data(self, log_dict: dict):
        # Always log the acceptance.
        if hasattr(self.state, "sampler_state"):

            # vstate
            chain_name = (
                self.importance_sampling_distribution.name
                if self.importance_sampling_distribution
                else self.original_distribution.name
            )
            sampler_state = self.state.sampler_states[chain_name]
            acceptance = getattr(sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict[f"acceptance_state_{chain_name}"] = acceptance

            # tstate
            chain_name = (
                self.importance_sampling_target_distribution.name
                if self.importance_sampling_target_distribution
                else self.original_target_distribution.name
            )
            sampler_state = self.target.sampler_states[chain_name]
            acceptance = getattr(sampler_state, "acceptance", None)
            if acceptance is not None:
                log_dict[f"acceptance_target_{chain_name}"] = acceptance

        # Log the quadratic model if requested.
        if self.info is not None:
            log_dict["info"] = self.info

    @property
    def target(self) -> MCState:
        r"""
        The untransformed target state :math:`| \phi \rangle`.

        If a transformation is applied to the target state, it is stored in the attribute
        :attr:`U_target`.
        """
        return self._target

    @property
    def U_target(self) -> Optional[DiscreteJaxOperator]:
        r"""
        The optional operator :math:`\hat{U}` acting on the target state :math:`\phi`.

        The overall target state is therefore :math:`\hat{U} | \phi \rangle`.
        """
        return self._U_target

    @property
    def V_state(self) -> Optional[DiscreteJaxOperator]:
        r"""
        The optional operator :math:`\hat{V}` whose inverse acts on the variational
        state :math:`\psi`.

        The overall variational state is therefore :math:`\hat{V}^{-1} | \psi \rangle`.
        """
        return self._V_state

    @property
    def cv_coeff(self) -> float | None:
        r"""
        The control variate mixing coefficient.

        When minimising the infidelity, this should be set to -0.5.
        """
        return self._cv_coeff

    @property
    def resample_fraction(self) -> Optional[float]:
        r"""
        The fraction of samples to resample at each step.

        Standard approaach is to keep this to None, which is equivalent to 1.0.
        However, by setting this to a lower fraction, only some samples will be resampled
        and the computational cost will be reduced.
        """
        return self._resample_fraction

    @property
    def original_distribution(self):
        if self.V_state is None:
            return DefaultDistribution()
        else:
            return LinearOperatorStateDistribution(operator=self.V_state, name="Vpsi")

    @property
    def original_target_distribution(self):
        if self.U_target is None:
            return DefaultDistribution()
        else:
            return LinearOperatorStateDistribution(operator=self.U_target, name="Uphi")

    @property
    def importance_sampling_target_distribution(self):
        return self._importance_sampling_target_distribution
