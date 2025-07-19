from typing import Any, Callable, Optional
from abc import abstractmethod
from functools import partial
import flax.core as fcore
import warnings

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp

from netket import jax as nkjax
from netket.stats import statistics, mean as distributed_mean
from netket.optimizer.solver import cholesky
from netket.vqs import MCState, FullSumState
from netket.utils import timing, struct, module_version
from netket.utils.types import ScalarOrSchedule, Optimizer, Array, PyTree
from netket.jax._jacobian.default_mode import JacobianMode

from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)
from advanced_drivers._src.driver.ngd.sr_srt_common import sr, srt
from advanced_drivers._src.driver.ngd.srt_onthefly import srt_onthefly
from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)

ApplyFun = Callable[[PyTree, Array], Array]
DerivativesArgs = tuple[ApplyFun, PyTree, Array, Array, tuple[Any, ...]]
KernelFun = Callable[[PyTree, Array], DerivativesArgs]


@jax.jit
def _flatten_samples(x):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(x, 0, x.ndim - 1)


class AbstractNGDDriver(AbstractVariationalDriver):
    r"""
    Abstract class for Natural Gradient Descent (NGD) drivers. This class is not meant to be used
    directly, but to be subclassed by specific NGD drivers. It provides the basic structure for
    the optimization loop, and the interface to the NGD solvers.

    The main method to be implemented by subclasses is `_get_local_estimators`, which should return
    the `local_gradient` and `local_loss` estimators.

    The class supports both the standard formulation of NGD and the kernel formulation. The two formulations
    compute the parameter updates as follows:

    - The standard formulation computes the updates as:

    .. math::
        \delta \theta = \tau (X^TX + \lambda \mathbb{I}_{N_P})^{-1} X^T f,

    where :math:`X \in R^{N_s \times N_p}` is the Jacobian of the log-wavefunction, with :math:`N_p` the number of parameters
    and :math:`N_s` the number of samples. The vector :math:`f` is the centered local estimator, corresponding, in the code, to ``local_gradient``.

    - The kernel formulation computes the updates as:

    .. math::
        \delta \theta = \tau X^T(XX^T + \lambda \mathbb{I}_{2N_s})^{-1} f,

    The matrix inversion is performed using a linear solver, which can be specified by the user.
    The regularization parameter :math:`\lambda` is the `diag_shift` parameter.
    The updates are then applied to the parameters using the `optimizer` which in general should be `optax.sgd`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.

    When `momentum` is used, this driver implements the SPRING optimizer in
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
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

    _mode: str = struct.field(serialize=False)
    _chunk_size_bwd: Optional[int] = struct.field(serialize=False)
    _use_ntk: bool = struct.field(serialize=False)
    _on_the_fly: bool = struct.field(serialize=False)
    _collect_quadratic_model: bool = struct.field(serialize=False)
    _collect_gradient_statistics: bool = struct.field(serialize=False)
    _linear_solver_fn: Any = struct.field(serialize=False)

    _importance_sampling_distribution: AbstractDistribution = struct.field(
        serialize=False
    )

    # Internal things cached
    _unravel_params_fn: Any = struct.field(serialize=False)

    # Serialized state
    _old_updates: PyTree = None
    _dp: PyTree = struct.field(serialize=False)
    info: Optional[Any] = None
    """
    PyTree to pass on information from the solver,e.g, the quadratic model.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        importance_sampling_distribution: AbstractDistribution | None = None,
        diag_shift: ScalarOrSchedule,
        proj_reg: Optional[ScalarOrSchedule] = None,
        momentum: Optional[ScalarOrSchedule] = None,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        variational_state: MCState = None,
        chunk_size_bwd: Optional[int] = None,
        collect_quadratic_model: bool = False,
        collect_gradient_statistics: bool = False,
        mode: Optional[JacobianMode] = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = None,
        minimized_quantity_name: str = "Loss",
    ):
        r"""
        Initialize the driver.

        Args:
            optimizer: The optimizer to use for the parameter updates.
            diag_shift: The regularization parameter :math:`\lambda` for the NGD solver.
            proj_reg: The regularization parameter for the projection of the updates.
            momentum: The momentum parameter for the optimizer.
            linear_solver_fn: The linear solver function to use for the NGD solver.
            mode: The mode used to compute the jacobian of the variational state.
                Can be `'real'` or `'complex'`.
            on_the_fly: Whether to compute the QGT or NTK using lazy evaluation methods.
                This usually requires less memory.
            variational_state: The variational state to optimize.
            chunk_size_bwd: The number of rows of the NTK evaluated in a single sweep.
            collect_quadratic_model: Whether to collect the quadratic model of the loss.
            collect_gradient_statistics: Whether to collect the statistics (mean and variance) of the gradient.
            use_ntk: Wheter to compute the updates using the Neural Tangent Kernel (NTK)
            instead of the Quantum Geometric Tensor (QGT).
            minimized_quantity_name: The name of the minimized quantity.
        """
        super().__init__(
            variational_state,
            optimizer,
            minimized_quantity_name=minimized_quantity_name,
        )
        if isinstance(variational_state, FullSumState):
            raise TypeError(
                "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
            )

        self._importance_sampling_distribution = importance_sampling_distribution

        if on_the_fly is None:
            if use_ntk:
                on_the_fly = True
            else:
                on_the_fly = False

        self.diag_shift = diag_shift
        self.proj_reg = proj_reg
        self.momentum = momentum

        self.chunk_size_bwd = chunk_size_bwd
        self._use_ntk = use_ntk
        self.mode = mode
        self.on_the_fly = on_the_fly

        self.collect_quadratic_model = collect_quadratic_model
        self.collect_gradient_statistics = collect_gradient_statistics

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

        # initialise the sampler
        _initialise_distribution(
            self.state,
            self.original_distribution,
            self.importance_sampling_distribution,
            chain_name=None,
        )

    @timing.timed
    def local_estimators(
        self,
    ) -> tuple[Array, Array]:
        r"""
        Compute the per-sample (local) estimators for the gradient and loss.
        The two match when minimising the energy, but can differ, as is the case of some Infidelity estimators.

        .. note::

            This method uses the log-wavefunction, the variables and the samples obtained by calling the method
            `_get_local_estimators_kernel_args`. This method should be implemented by subclasses.

        Returns:
                The method should return 3 jax arrays of shape `(Ns,)`:  `local_grad`, `local_loss`, and `weights`.
                `local_loss` should be such that, once multiplied component-wise by the weights, its mean equals the expectation value of the loss.
                `local_grad` should be such that, once multiplied component-wise by the weights and by a component of the jacobian, its mean equals the expectation value of the selected component of the gradient of the loss.
        """
        afun, variables, samples, weights, extra_args = self._prepare_derivatives()
        local_grad, local_loss = self._kernel(
            afun,
            variables,
            samples,
            *extra_args,
        )
        return local_grad, local_loss, weights

    @property
    def importance_sampling_distribution(self) -> AbstractDistribution:
        r"""
        The distribution from which we sample the local gradient estimators for the QGT/NTK of
        the natural gradient.
        """
        return self._importance_sampling_distribution

    @property
    @abstractmethod
    def _kernel(self) -> KernelFun:
        r"""
        The kernel function to compute the local estimators.

        .. note::

            This method should be implemented by subclasses.

        This resulting kernel function should be called as follows:

        .. code-block:: python

            afun, variables, σ, extra_args = self._get_local_estimators_kernel_args()
            local_grad, local_loss = self._kernel(afun, variables, σ, *extra_args)

        Returns:
            A function that takes the log-wavefunction, the variables, the samples and any extra argument,
            and returns the local gradient and the local loss.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def original_distribution(self) -> ApplyFun:
        r"""
        The orginal distribution over which the expectations of the local estimators are evaluated.

        .. note::

            This method should be implemented by subclasses.

        This distribution will be used to compute the resampling weights if `importance_sampling_distribution`
        when not doing importance sampling.

        Returns:
            A function that takes the variables and the samples and returns the log-amplitude of the distribution.
        """
        raise NotImplementedError()

    @timing.timed
    def _prepare_derivatives(self) -> DerivativesArgs:
        r"""
        Prepare the function and the samples for the computation of the jacobian, the neural tangent kernel,
        the vjp or jvp.

        This method difers from `_get_local_estimators_kernel_args` in that it is not used to compute the local
        estimators, but to compute the NGD update. In general, this should be the same function as the ones
        used to compute the local estimator of the gradient, but not necessarily.

        .. note::

            This method should be implemented by subclasses.

        Returns:
            A tuple containing the function, the parameters, the model state, the samples and the importance weights to be fed to the jacobian, the neural tangent kernel, the vjp or jvp.
        """
        raise NotImplementedError()

    @timing.timed
    def compute_loss_and_update(self):
        # equivalent to using `local_estimator`
        afun, variables, samples, weights, extra_args = self._prepare_derivatives()
        local_grad, local_loss = self._kernel(
            afun,
            variables,
            samples,
            *extra_args,
        )

        self._loss_stats = statistics(local_loss * weights)

        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)

        model_state, params = fcore.pop(variables, "params")
        self._dp, self._old_updates, self.info = self.update_fn(
            afun,
            local_grad,
            params,
            model_state,
            samples,
            importance_sampling_weights=weights,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver_fn,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd,
            collect_quadratic_model=self.collect_quadratic_model,
            collect_gradient_statistics=self.collect_gradient_statistics,
        )

        return self._loss_stats, self._dp

    @timing.timed
    def _log_additional_data(self, log_dict: dict):
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
        Whether to compute the Jacobian in full (False) or evaluate one row at the time as if we were using vjp (True).
        This usually requires less memory.
        """
        return self._on_the_fly

    @on_the_fly.setter
    def on_the_fly(self, value: bool):
        if value and not self.use_ntk:
            raise ValueError(
                """
                `onthefly` is only supported when `use_ntk=True`.
                We plan to support this mode for the standard NGD in the future.
                In the meantime, use a standard VMC+SR QGTOnTheFly.
                """
            )
        self._on_the_fly = value

    @property
    def use_ntk(self) -> bool:
        r"""
        Whether to use the Neural Tangent Kernel (NTK) instead of the Quantum Geometric Tensor (QGT) to compute the update.
        """
        return self._use_ntk

    @property
    def update_fn(self) -> Callable:
        """Returns the function to compute the NGD update based on the evaluation mode."""
        if self.use_ntk:
            if self.on_the_fly:
                return srt_onthefly
            else:
                return srt
        else:
            if self.on_the_fly:
                raise NotImplementedError
            else:
                return sr

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

    @property
    def collect_quadratic_model(self) -> bool:
        r"""
        Whether to collect the quantities required to compute the quadratic model of the loss.

        These quantities, `linear_term` and `qudratic_term` are collected in the `info` attribute of the driver.
        They are computed during the estimation of the parameter updates by the functions `_compute_quadratic_model_sr`
        and `_compute_quadratic_model_srt`.

        The quadratic model is used to estimate if the update is small enough for the
        quadratic approximation on which NGD is based upon remains valid after the update.
        """
        return self._collect_quadratic_model

    @collect_quadratic_model.setter
    def collect_quadratic_model(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("collect_quadratic_model must be a boolean")
        self._collect_quadratic_model = value

    @property
    def collect_gradient_statistics(self) -> bool:
        r"""
        Whether to collect the mean and variance of the gradient.

        These quantities, `grad_mean` and `grad_variance` are collected in the `info` attribute of the driver.
        """
        return self._collect_gradient_statistics

    @collect_gradient_statistics.setter
    def collect_gradient_statistics(self, value: bool):
        if not isinstance(value, bool):
            raise TypeError("collect_gradient_statistics must be a boolean")
        if self.on_the_fly and value:
            warnings.warn(
                "The gradient statistics can only be computed with `on_the_fly=False`. Setting `collect_gradient_statistics=False`."
            )
            value = False
        self._collect_gradient_statistics = value

    def roofline_experimental(self, verbose: bool = True):
        """
        Experimental method to compute the roofline model of the NGD driver.

        This is not implemented yet, but it will be used to compute the roofline model of the NGD driver.
        """
        # equivalent to using `local_estimator`
        afun, variables, samples, weights, extra_args = self._prepare_derivatives()
        local_grad, local_loss = self._kernel(
            afun,
            variables,
            samples,
            *extra_args,
        )

        self._loss_stats = statistics(local_loss * weights)

        diag_shift = self.diag_shift
        proj_reg = self.proj_reg
        momentum = self.momentum
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)
        if callable(proj_reg):
            proj_reg = proj_reg(self.step_count)
        if callable(momentum):
            momentum = momentum(self.step_count)
        model_state, params = fcore.pop(variables, "params")

        # Compile and get info
        update_fn = self.update_fn
        kwargs = {}
        if isinstance(update_fn, partial):
            kwargs = update_fn.keywords
            update_fn = update_fn.func

        update_fn_lowered = update_fn.lower(
            afun,
            local_grad,
            params,
            model_state,
            samples,
            importance_sampling_weights=weights,
            diag_shift=diag_shift,
            solver_fn=self._linear_solver_fn,
            mode=self.mode,
            proj_reg=proj_reg,
            momentum=momentum,
            old_updates=self._old_updates,
            chunk_size=self.chunk_size_bwd,
            collect_quadratic_model=self.collect_quadratic_model,
            collect_gradient_statistics=self.collect_gradient_statistics,
            **kwargs,
        )
        update_fn_compiled = update_fn_lowered.compile()

        args = (local_grad, params, model_state, samples, weights, diag_shift, momentum)

        def myf(
            local_grad,
            params,
            model_state,
            samples,
            weights,
            diag_shift,
            momentum,
        ):
            return self.update_fn(
                afun,
                local_grad,
                params,
                model_state,
                samples,
                importance_sampling_weights=weights,
                diag_shift=diag_shift,
                solver_fn=self._linear_solver_fn,
                mode=self.mode,
                proj_reg=proj_reg,
                momentum=momentum,
                old_updates=self._old_updates,
                chunk_size=self.chunk_size_bwd,
                collect_quadratic_model=self.collect_quadratic_model,
                collect_gradient_statistics=self.collect_gradient_statistics,
            )

        from jax.experimental import roofline

        _, rfline = roofline.roofline(myf)(*args)
        cost_anal = update_fn_compiled.cost_analysis()
        mem_anal = update_fn_compiled.memory_analysis()
        msg = f"""
        Returning the Roofline model and compiled information.
        Check the second argument .cost_analysis and .memory_analysis for details
        Memory analysis:
            - argument sizes : {mem_anal.argument_size_in_bytes/1024**3} GB
            - output sizes   : {mem_anal.output_size_in_bytes/1024**3} GB
            - temp size      : {mem_anal.temp_size_in_bytes/1024**3} GB
            - alias size     : {mem_anal.alias_size_in_bytes/1024**3} GB
        Cost Analysis:
            - GFLOPs         : {cost_anal['flops']/1024**3} TFLOPs
        """
        if verbose:
            print(msg)

        return rfline, update_fn_compiled


def _get_samples_and_weights(
    state,
    original_distribution,
    importance_sampling_distribution,
    chunk_size,
    chain_name=None,
):
    """
    Common function to sample and compute the weights for the samples of a function.

    chain name overrides the name of the markov chain sampled.
    """
    log_prob_p_fun, variables_p = original_distribution(
        state._apply_fun, state.variables
    )

    if importance_sampling_distribution is None:
        samples, log_probabilities_p = state.samples_distribution(
            log_prob_p_fun,
            variables=variables_p,
            chain_name=(
                chain_name if chain_name is not None else original_distribution.name
            ),
            return_log_probabilities=True,
            chunk_size=chunk_size,
        )
        samples = _flatten_samples(samples)
        weights = _jnp_ones(
            (samples.shape[0],),
            dtype=log_probabilities_p.dtype,
            sharding=jax.sharding.PositionalSharding(jax.devices()).reshape(-1),
        )
    else:
        log_prob_q_fun, variables_q = importance_sampling_distribution(
            state._apply_fun, state.variables
        )
        samples, log_probabilities_q = state.samples_distribution(
            log_prob_q_fun,
            variables=variables_q,
            chain_name=(
                chain_name
                if chain_name is not None
                else importance_sampling_distribution.name
            ),
            return_log_probabilities=True,
            chunk_size=chunk_size,
        )
        samples = _flatten_samples(samples)
        weights = _compute_weights(
            samples,
            afun_p=log_prob_p_fun,
            vars_p=variables_p,
            machine_pow_p=state.sampler.machine_pow,
            log_prob_q=log_probabilities_q.reshape(-1),
            chunk_size=chunk_size,
        )
        weights = weights / distributed_mean(weights)

    return log_prob_p_fun, variables_p, samples, weights


def _initialise_distribution(
    state,
    original_distribution,
    importance_sampling_distribution,
    chain_name=None,
):
    """
    Initialise a distribution
    """
    distribution = (
        original_distribution
        if importance_sampling_distribution is None
        else importance_sampling_distribution
    )
    chain_name = chain_name if chain_name is not None else distribution.name

    if chain_name not in state.sampler_states:
        log_prob_p_fun, variables_p = distribution(state._apply_fun, state.variables)
        state.init_sampler_distribution(
            log_prob_p_fun,
            variables=variables_p,
            chain_name=chain_name,
        )


# Function to compute the importance weights
@partial(jax.jit, static_argnames=("afun_p", "afun_q", "chunk_size"))
def _compute_weights(
    samples,
    *,
    afun_q=None,
    vars_q=None,
    log_prob_q=None,
    machine_pow_q=2,
    afun_p=None,
    vars_p=None,
    log_prob_p=None,
    machine_pow_p=2,
    chunk_size: int | None = None,
):
    """
    Compute the importance weights for the samples, assuming the distributions are
    log-sqrt of what is sampled.

    You can pass either the log probabilities themselves of p and/or q, or the functions
    that compute them, and the variables of the distributions.

    A valid call signature is:

    .. code-block:: python
        _compute_weights(samples, log_prob_q=log_prob_q, afun_p=afun_p, vars_p=vars_p)

    Args:
        samples: The samples to compute the weights for.
        afun_q: The function that computes the log probability of the sampling distribution.
        vars_q: The variables of the sampling distribution.
        log_prob_q: The log probability of the sampling distribution.
        afun_p: The function that computes the log probability of the original distribution.
        vars_p: The variables of the original distribution.
        log_prob_p: The log probability of the original distribution.
        chunk_size: The chunk size to use for the backward pass (jacobian or vjp evaluation).
    """
    if log_prob_q is not None:
        assert log_prob_q.shape == samples.shape[:-1]
    elif afun_q is not None:
        afun_q_ = nkjax.apply_chunked(partial(afun_q, vars_q), chunk_size=chunk_size)
        log_prob_q = machine_pow_q * afun_q_(samples).real
    else:
        raise ValueError(
            "Either log_prob_q or afun_q must be provided to compute the weights."
        )

    if log_prob_p is not None:
        assert log_prob_p.shape == log_prob_q.shape
    elif afun_p is not None:
        afun_p_ = nkjax.apply_chunked(partial(afun_p, vars_p), chunk_size=chunk_size)
        log_prob_p = machine_pow_p * afun_p_(samples).real
    else:
        raise ValueError(
            "Either log_prob_p or afun_p must be provided to compute the weights."
        )

    return jnp.exp(log_prob_p - log_prob_q)


# TODO : when we require jax >= 0.6.0, we can use jnp.ones directly
def _jnp_ones(
    shape,
    *,
    dtype=jnp.float_,
    sharding=None,
):
    r"""
    Create an array of ones with the given shape and dtype with a sharding.
    Compatibility for jax < 0.6.0.

    Args:
        shape: The shape of the array.
        dtype: The dtype of the array.
        sharding: The sharding of the array.

    Returns:
        An array of ones with the given shape and dtype.
    """
    if module_version(jax) >= (0, 6, 0):
        return jnp.ones(shape, dtype=dtype, out_sharding=sharding)
    else:
        arr = jnp.ones(shape, dtype=dtype)
        return jax.lax.with_sharding_constraint(arr, sharding)
