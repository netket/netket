from typing import Callable, Optional, Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import flax.core as fcore

from netket import stats as nkstats
from netket.jax import (
    tree_ishomogeneous,
    jacobian_default_mode,
)
from netket.optimizer.solver import cholesky
from netket.operator import AbstractOperator
from netket.utils import struct
from netket.utils.types import (
    ScalarOrSchedule,
    Optimizer,
    PyTree,
    Array,
)
from netket.vqs import FullSumState

from netket_pro.utils.sampling_Ustate import _lazy_apply_UV_to_afun
from netket_pro._src.operator.jax_utils import to_jax_operator

from advanced_drivers.driver import AbstractVariationalDriver

from netket_pro._src import distributed as distributed

from advanced_drivers._src.driver.ngd.sr_srt_common import sr, srt


class InfidelityOptimizerNG_FS(AbstractVariationalDriver):
    """
    Full-summation driver implementation for the Infidelity NGD optimization.

    Roughly equivalent to :class:`advanced_drivers.driver.InfidelityOptimizerNG`.
    """

    _target: FullSumState = struct.field(pytree_node=False, serialize=False)
    _U_target: Optional[AbstractOperator] = struct.field(
        pytree_node=False, serialize=False
    )
    _V_state: Optional[AbstractOperator] = struct.field(
        pytree_node=False, serialize=False
    )
    _linear_solver_fn: Any = struct.field(serialize=False)

    _unravel_params_fn: Callable[[Array], PyTree] = struct.field(
        pytree_node=False, serialize=False
    )
    _params_structure: PyTree = struct.field(pytree_node=False, serialize=False)
    _jacobian_mode: str = struct.field(pytree_node=False, serialize=False)
    _old_updates = struct.field(pytree_node=False, serialize=False, default=None)
    info: PyTree = struct.field(pytree_node=False, serialize=False)
    diag_shift: ScalarOrSchedule = struct.field(pytree_node=False, serialize=False)
    collect_quadratic_model: bool = struct.field(pytree_node=False, serialize=False)
    use_ntk: bool = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        target_state: FullSumState,
        optimizer: Optimizer,
        variational_state: FullSumState,
        *,
        diag_shift: ScalarOrSchedule,
        linear_solver_fn: Callable[[Array, Array], Array] = cholesky,
        jacobian_mode: Optional[str] = None,
        U: Optional[AbstractOperator] = None,
        V: Optional[AbstractOperator] = None,
        collect_quadratic_model: bool = False,
        use_ntk: bool = False,
    ):
        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        if target_state is variational_state:
            raise ValueError(
                "Target state and variational_state must be two different objects."
            )

        self._params_structure = jax.tree_util.tree_map(
            lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), self.state.parameters
        )
        if not tree_ishomogeneous(self._params_structure):
            raise ValueError(
                "SRt only supports neural networks with all real or all complex "
                "parameters. Hybrid structures are not yet supported."
            )

        self._target = target_state
        self._U_target = to_jax_operator(U) if U is not None else None
        self._V_state = to_jax_operator(V) if V is not None else None

        self.info: PyTree = None

        self.diag_shift = diag_shift
        self.jacobian_mode = jacobian_mode
        self._linear_solver_fn = linear_solver_fn

        _, unravel_params_fn = ravel_pytree(self.state.parameters)
        self._unravel_params_fn = jax.jit(unravel_params_fn)
        self._old_updates: PyTree = None

        self.collect_quadratic_model = collect_quadratic_model
        self.use_ntk = use_ntk

    def reset_step(self, hard=False):
        pass

    def compute_loss_and_update(self):
        samples = self.state._all_states

        afun, vars, _ = _lazy_apply_UV_to_afun(
            self.state, self.V_state, extra_hash_data="V"
        )
        model_state, parameters = fcore.pop(vars, "params")

        ψ = (
            self.V_state.to_sparse() @ self.state.to_array()
            if self.V_state is not None
            else self.state.to_array()
        )
        pdf_ψ = jnp.abs(ψ) ** 2

        ϕ = (
            self.U_target.to_sparse() @ self.target.to_array()
            if self.U_target is not None
            else self.target.to_array()
        )
        ϕ = ϕ / jnp.linalg.norm(ϕ)

        pdf_ϕ = jnp.abs(ϕ) ** 2

        E = jnp.sum(pdf_ϕ * ψ / ϕ)
        floc = 1 - (ϕ / ψ * E)

        loss_mean = jnp.sum(pdf_ψ * floc)
        self._loss_stats = nkstats.Stats(
            mean=loss_mean,
            variance=0.0,
            error_of_mean=0.0,
        )

        diag_shift = self.diag_shift
        if callable(diag_shift):
            diag_shift = diag_shift(self.step_count)

        sr_or_srt = srt if self.use_ntk else sr

        self._dp, self._old_updates, info = sr_or_srt(
            afun,
            floc,
            parameters,
            model_state,
            samples,
            diag_shift=diag_shift,
            old_updates=self._old_updates,
            solver_fn=self._linear_solver_fn,
            mode=self.jacobian_mode,
            collect_quadratic_model=self.collect_quadratic_model,
            pdf=pdf_ψ,
        )
        if self.info is None:
            self.info = info
        else:
            self.info.update(info)
        return self._loss_stats, self._dp

    @property
    def jacobian_mode(self) -> str:
        """
        The mode used to compute the jacobian of the variational state. Can be `'real'`
        or `'complex'`.

        Real mode truncates imaginary part of the wavefunction, while `complex` does not.
        This internally uses :func:`netket.jax.jacobian`. See that function for a more
        complete documentation.
        """
        return self._jacobian_mode

    @jacobian_mode.setter
    def jacobian_mode(self, mode: Optional[str]):
        if mode is None:
            mode = jacobian_default_mode(
                self.state._apply_fun,
                self.state.parameters,
                self.state.model_state,
                self.state.hilbert.random_state(jax.random.key(1), 3),
                warn=False,
            )

        if mode not in ["complex", "real"]:
            raise ValueError(
                "`jacobian_mode` only supports 'real' for real-valued wavefunctions and"
                "'complex'.\n\n"
                "`holomorphic` is not yet supported, but could be contributed in the future."
            )
        self._jacobian_mode = mode

    @property
    def U_target(self) -> Optional[AbstractOperator]:
        """
        The operator U used to compute the target state.
        """
        return self._U_target

    @property
    def V_state(self) -> Optional[AbstractOperator]:
        """
        The operator U used to compute the target state.
        """
        return self._V_state

    @property
    def infidelity(self) -> nkstats.Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    @property
    def target(self) -> FullSumState:
        """
        The target state of the driver.
        """
        return self._target

    def __repr__(self):
        return (
            "Infidelity_SR("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
