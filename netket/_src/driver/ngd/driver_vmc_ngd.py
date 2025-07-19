from typing import Callable, Optional
from functools import partial
import jax

from netket.optimizer.solver import cholesky
from netket.utils.types import Array, Optimizer, ScalarOrSchedule
from netket.operator import AbstractOperator
from netket.utils import struct, timing
from netket.vqs.mc import MCState, get_local_kernel, get_local_kernel_arguments
from netket.jax._jacobian.default_mode import JacobianMode
from netket import jax as nkjax

from netket_pro._src import distributed as distributed

from advanced_drivers._src.distribution.abstract_distribution import (
    AbstractDistribution,
)
from advanced_drivers._src.distribution.default import (
    DefaultDistribution,
)
from advanced_drivers._src.driver.ngd.driver_abstract_ngd import (
    AbstractNGDDriver,
    KernelFun,
    DerivativesArgs,
    _get_samples_and_weights,
)


class VMC_NG(AbstractNGDDriver):
    r"""
    Energy minimization using Variational Monte Carlo (VMC) and Stochastic Reconfiguration (SR)
    with or without its kernel formulation. The two approaches lead to *exactly* the same parameter
    updates. In the kernel SR framework, the updates of the parameters can be written as:

    .. math::
        \delta \theta = \tau X(X^TX + \lambda \mathbb{I}_{2M})^{-1} f,

    where :math:`X \in R^{P \times 2M}` is the concatenation of the real and imaginary part
    of the centered Jacobian, with P the number of parameters and M the number of samples.
    The vector f is the concatenation of the real and imaginary part of the centered local
    energy. Note that, to compute the updates, it is sufficient to invert an :math:`M\times M` matrix
    instead of a :math:`P\times P` one. As a consequence, this formulation is useful
    in the typical deep learning regime where :math:`P \gg M`.

    See `R.Rende, L.L.Viteritti, L.Bardone, F.Becca and S.Goldt <https://arxiv.org/abs/2310.05715>`_
    for a detailed description of the derivation. A similar result can be obtained by minimizing the
    Fubini-Study distance with a specific constrain, see `A.Chen and M.Heyl <https://arxiv.org/abs/2302.01941>`_
    for details.

    When `momentum` is used, this driver implements the SPRING optimizer in
    `G.Goldshlager, N.Abrahamsen and L.Lin <https://arxiv.org/abs/2401.10190>`_
    to accumulate previous updates for better approximation of the exact SR with
    no significant performance penalty.
    """

    _ham: AbstractOperator = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        hamiltonian: AbstractOperator,
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
        mode: Optional[JacobianMode] = None,
        use_ntk: bool = False,
        on_the_fly: bool | None = None,
    ):
        r"""
        Initialize the driver.

        Args:
            hamiltonian: The Hamiltonian of the system.
            optimizer: Determines how optimization steps are performed given the bare energy gradient.
            diag_shift: The diagonal shift of the curvature matrix.
            proj_reg: Weight before the matrix `1/N_samples \\bm{1} \\bm{1}^T` used to regularize the linear solver in SPRING.
            momentum: Momentum used to accumulate updates in SPRING.
            linear_solver_fn: Callable to solve the linear problem associated to the updates of the parameters.
            mode: The mode used to compute the jacobian or vjp of the variational state.
                Can be `'real'` or `'complex'` (defaults to the dtype of the output of the model).
                `real` can be used for real wavefunctions with a sign to further reduce the computational costs.
            on_the_fly: Whether to compute the QGT or NTK matrix without evaluating the full jacobian. Defaults to True.
                This ususally lowers the memory requirement and is necessary for large calculations.
            use_ntk: Whether to use the NTK instead of the QGT for the computation of the updates.
            variational_state: The :class:`netket.vqs.MCState` to be optimised. Other variational states are not supported.
            chunk_size_bwd: The chunk size to use for the backward pass (jacobian or vjp evaluation).
            collect_quadratic_model: Whether to collect the quadratic model. The quantities collected are the linear and quadratic term in the approximation of the loss function. They are stored in the info dictionary of the driver.

        Returns:
            The new parameters, the old updates, and the info dictionary.
        """
        self._ham = hamiltonian.collect()  # type: AbstractOperator
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
            minimized_quantity_name="Energy",
        )

    @timing.timed
    def _prepare_derivatives(self) -> DerivativesArgs:
        r"""
        Prepare the function and the samples for the computation of the jacobian, the neural tangent kernel, the vjp or jvp.
        If relevant, computes the importance weights associated to the sampling distribution.
        Returns:
            A tuple containing the function, the parameters, the model state, the samples and the importance weights to be fed
            to the jacobian, the neural tangent kernel, the vjp or jvp.
        """
        log_prob_p_fun, variables_p, samples, weights = _get_samples_and_weights(
            state=self.state,
            original_distribution=self.original_distribution,
            importance_sampling_distribution=self.importance_sampling_distribution,
            chunk_size=self.state.chunk_size,
        )

        # TODO: `get_local_kernel_arguments` also samples the psi distribution.
        # Since we do not need the samples from here, we should write a custom kernel_args to only retrieve the extra_args
        _, extra_args = get_local_kernel_arguments(self.state, self._ham)

        return log_prob_p_fun, variables_p, samples, weights, (extra_args,)

    @property
    def _kernel(self) -> KernelFun:
        chunk_size = self.state.chunk_size

        if chunk_size is None:
            kernel = get_local_kernel(self.state, self._ham)
        else:
            kernel = nkjax.HashablePartial(
                get_local_kernel(self.state, self._ham, chunk_size),
                chunk_size=chunk_size,
            )
        return nkjax.HashablePartial(_vmc_local_kernel, kernel)

    @property
    def original_distribution(self):
        return DefaultDistribution()


# The original kernel from netket follows a different signature, so we must wrap
# the kernel to make it compatible with the new signature.
@partial(jax.jit, static_argnames=("kernel", "logpsi"))
def _vmc_local_kernel(
    kernel,
    logpsi,
    vars,
    samples,
    operator,
):
    local_energies = kernel(logpsi, vars, samples, operator)
    return local_energies, local_energies
