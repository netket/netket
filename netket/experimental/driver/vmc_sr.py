from typing import Any
from collections.abc import Callable

import jax
from jax.flatten_util import ravel_pytree

from netket import jax as nkjax
from netket.optimizer.solver import cholesky
from netket.vqs import MCState, FullSumState
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


ApplyFun = Callable[[PyTree, Array], Array]
KernelArgs = tuple[ApplyFun, PyTree, Array, tuple[Any, ...]]
KernelFun = Callable[[PyTree, Array], KernelArgs]
DeriativesArgs = tuple[ApplyFun, PyTree, PyTree, Array]


@jax.jit
def _flatten_samples(x):
    # return x.reshape(-1, x.shape[-1])
    return jax.lax.collapse(x, 0, x.ndim - 1)


def VMC_SRt(
    hamiltonian: AbstractOperator,
    optimizer: Optimizer,
    *,
    diag_shift: ScalarOrSchedule,
    linear_solver_fn: Callable[[jax.Array, jax.Array], jax.Array] = cholesky,
    mode: str | None = None,
    jacobian_mode: str | None = None,
    variational_state: MCState = None,
):
    if mode is None:
        mode = jacobian_mode
    elif mode is not None and jacobian_mode is not None:
        raise ValueError(
            "`jacobian_mode` is deprecated and renamed to `mode`. Just declare `mode`."
        )

    return VMC_SR(
        hamiltonian,
        optimizer,
        diag_shift=diag_shift,
        linear_solver_fn=linear_solver_fn,
        mode=mode,
        variational_state=variational_state,
        use_ntk=True,
        on_the_fly=False,
    )


class VMC_SR(AbstractVariationalDriver):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and **Stochastic Reconfiguration/Natural Gradient Descent**.
    This driver is mathematically equivalent to the standard :class:`netket.driver.VMC` with the
    preconditioner :class:`netket.optimizer.SR(solver=netket.optimizer.solvers.cholesky) <netket.optimizer.SR>`,
    but can easily switch between the standard and the kernel/minSR formulation of Natural Gradient Descent.

    - The standard formulation computes the updates as:

    .. math::
        \delta \theta = \tau (X^TX + \lambda \mathbb{I}_{N_P})^{-1} X^T E^{loc},

    where :math:`X \in R^{N_s \times N_p}` is the Jacobian of the log-wavefunction, with :math:`N_p` the number of parameters
    and :math:`N_s` the number of samples. The vector :math:`E^{loc}` is the centered local estimator for the local energies.

    - The kernel/minSR formulation computes the updates as:

    .. math::
        \delta \theta = \tau X^T(XX^T + \lambda \mathbb{I}_{2N_s})^{-1} E^{loc},

    The regularization parameter :math:`\lambda` is the `diag_shift` parameter of the driver, which can be
    a scalar or a schedule.
    The updates are then applied to the parameters using the `optimizer` which in general should be `optax.sgd`.


    Matrix Inversion
    ----------------

    The matrix inversion of both methods is performed using a linear solver, which can be specified by the user.
    This must be a function, the :code:`linear_solver_fun` argument, which has the following signature:

    .. code-block:: python

        linear_solver_fn(A: Matrix, b: vector) -> tuple[jax.Array[vector], dict]

    Where the vector is the solution and the dictionary may contain additional information about the solver or be None.
    The standard solver is based on the Cholesky decomposition :func:`~netket.optimizer.solver.cholesky`, but any other
    solver from `JAX <https://jax.readthedocs.io/en/latest/jax.experimental.linalg.html>`_, `netket solvers <dense-solvers>`_ or a
    custom-written one can be used.


    Natural Gradient Descent
    ------------------------

    Stochastic Reconfiguration is equivalent to the Natural Gradient Descent method introduced by
    `Amari 1998 <https://ieeexplore.ieee.org/abstract/document/6790500/>`_ in the context of neural network training,
    assuming that the *natural metric* of the space of wave-functions is the Fubini-Study metric. This was first
    studied by `Stokes et Al 2019 <https://arxiv.org/abs/1909.02108>`_ and called
    *quantum Natural Gradient Descent*.

    While stochastic reconfiguration has been heavily studied in the context of VMC, there is a vast literature
    in the Machine Learning community on the use of NGD, and tuning carefully the diag shift and the learning rate.

    A very good introduction to the mathematics of Information Geometry and NGD is found in
    `Bai et Al <https://arxiv.org/pdf/2202.06232>`_ and further studied in `Shrestha et Al 2022 <https://arxiv.org/pdf/2303.05473>`_.
    From the Physicist point of view, a good discussion on the choice of the metric function (QGT vs Fisher Matrix)
    is found in `Stokes et Al 2022 <https://arxiv.org/pdf/2203.14824>`_ (section 4 in particoular).
    For a comprehensive review of the method, we suggest the review by
    `Martens 2014 <https://arxiv.org/abs/1412.1193>`_.


    Momentum / SPRING
    -----------------
    When `momentum` is used, this driver implements the SPRING optimizer in
    `Goldshlager et Al. (2024) <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.

    `momentum` μ is a number between [0,1] that specifies the damping factor of
    the previous updates and works somewhat similarly to the beta parameter of ADAM.
    The difference is that rather than simply adding the damped previous update to the
    new update, SPRING uses the damped previous update to fill in the components of the
    SR direction that are not sampled by the current batch of walkers, resulting in a
    more accurate and less noisy estimate. Since SPRING only uses the previous update to
    fill in directions that are orthogonal to the current one, the maximum amplification of
    the step size in SPRING is :math:`A(\mu) = 1/\sqrt{1-μ^2}` rather than :math:`1/(1-μ)`.

    Thus the  amplification is at most a factor of :math:`A(0.9)=2.3` or
    :math:`A(0.99)=7.1`.
    ** Values that empirically work are around 0.8. **

    Some progress has been made on theoretically analyzing this parameter, in particular
    `Section 3 of Epperly et Al. <https://arxiv.org/pdf/2411.19877>`_ demonstrates (albeit
    in a significantly  simplified linear least-squares setting) that SPRING can be interpreted
    as iteratively estimating a regularized SR direction, with the amount of regularization
    proportional to  the value of 1-momentum. Additional insights regarding the behavior of
    some SPRING-like algorithms, albeit still in the linear least-squares setting, are presented in
    `Goldshlager et Al. (2025) <https://arxiv.org/pdf/2502.00882>`_ .


    Implementation details
    ------------------------
    The kernel-trick/NTK based implementation can run with both a direct calculation of the jacobian
    (`on_the_fly=False`) or with a lazy evaluation of the NTK (`on_the_fly=True`). The latter is more
    computationally efficient for networks that reuse the parameters many times for every forward pass
    (convolutions, attention layers, but not dense layers...) and generally uses less memory.

    However, the on the fly implementation relies on some JAX compiler behaviour, so it might at times
    have worse performance. We suggest you check on your specific model. For a more detailed explanation
    of the on-the-fly implementation of the NTK, we refer to `Novak et Al 2022 <https://arxiv.org/pdf/2206.08720>`_.
    The algorithm netket uses is the layer-wise jacobian contraction method (sec 3.2) of the manuscript.

    The default choice is to use the ``on_the_fly=True`` mode.

    References
    ----------
    - Stochastic Reconfiguration was originally introduced in the QMC field by `Sorella <https://arxiv.org/abs/cond-mat/9803107>`_.
      The method was later shown to be equivalent to the Natural Gradient Descent method introduced by
      `Amari <https://ieeexplore.ieee.org/abstract/document/6790500/>`_ for the Fubini-Study metric.

    - The *kernel trick* which makes NGD/SR feasible in the large-parameter count limit was originally introduced
      to the field of NQS by `Chen & Heyl <https://arxiv.org/abs/2302.01941>`_ under the name of `minSR`.
      `Rende & Al <https://arxiv.org/abs/2310.05715>`_ proposed a simpler derivation in terms of the Kernel trick.

      It's interesting to note that those tricks were first mentioned by `Ren & Goldfarb <https://arxiv.org/abs/1906.02353>`_
      in the ML community.

    - When using Momentum you should cite `G.Goldshlager et Al. (2024) <https://arxiv.org/abs/2401.10190>`_.
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
        hamiltonian: AbstractOperator,
        optimizer: Optimizer,
        *,
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
            hamiltonian: The Hamiltonian of which the ground-state is to be found.
            optimizer: The optimizer to use for the parameter updates. To perform proper
                SR/NGD optimization this should be an instance of `optax.sgd`, but can be
                any other optimizer if you are brave.
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
        if isinstance(variational_state, FullSumState):
            raise TypeError(
                "NGD drivers do not support FullSumState. Please use 'standard' drivers with SR."
            )
        super().__init__(variational_state, optimizer, minimized_quantity_name="Energy")

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

        self._ham = hamiltonian

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

        # Compute the local energy estimator and average Energy
        local_energies = self.state.local_estimators(self._ham)
        self._loss_stats = nkstats.statistics(local_energies)

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
