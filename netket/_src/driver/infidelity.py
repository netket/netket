from typing import Optional, Callable
from netket.stats import Stats
from netket.utils import struct
from netket.vqs import VariationalState
from netket.operator import AbstractOperator
from netket.callbacks import ConvergenceStopping
from netket.optimizer import (
    identity_preconditioner,
    PreconditionerT,
)

from netket_pro.infidelity import InfidelityOperator

from advanced_drivers._src.utils.itertools import to_iterable
from advanced_drivers._src.driver.abstract_variational_driver import (
    AbstractVariationalDriver,
)


class InfidelityOptimizer(AbstractVariationalDriver):
    r"""
    Adriver training the state to match the target state, on which a Unitary
    might act.

    The target state is either `math`:\ket{\psi}` or `math`:\hat{U}\ket{\psi}`
    depending on the provided inputs.

    .. warning::

        This driver exists for NOT using Natural Gradient/SR. If you wish to use NGD, refer rather to the other drivers.

    """

    _cv: float | None = struct.field(serialize=False)
    _preconditioner: PreconditionerT = identity_preconditioner
    _I_op: InfidelityOperator = struct.field(pytree_node=False, serialize=False)

    def __init__(
        self,
        target_state: VariationalState,
        optimizer,
        *,
        variational_state: VariationalState,
        U: Optional[AbstractOperator] = None,
        U_dagger: Optional[AbstractOperator] = None,
        V: Optional[AbstractOperator] = None,
        preconditioner: PreconditionerT = identity_preconditioner,
        is_unitary: bool = False,
        sample_Uphi: bool = True,
        cv_coeff: float = -0.5,
        resample_fraction: Optional[float] = None,
    ):
        r"""
        Operator I_op computing the infidelity I among two variational states
        :math:`|\psi\rangle` and :math:`|\phi\rangle` as:

        .. math::

            I = 1 - \frac{|⟨\Psi|\Phi⟩|^2 }{ ⟨\Psi|\Psi⟩ ⟨\Phi|\Phi⟩ } = 1 - \frac{⟨\Psi|\hat{I}_{op}|\Psi⟩ }{ ⟨\Psi|\Psi⟩ }

        where:

        .. math::

            I_{op} = \frac {|\Phi\rangle\langle\Phi| }{ \langle\Phi|\Phi\rangle }

        The state :math:`|\phi\rangle` can be an autonomous state :math:`|\Phi\rangle = |\phi\rangle`
        or an operator :math:`U` applied to it, namely
        :math:`|\Phi\rangle  = U|\phi\rangle`. :math:`I_{op}` is defined by the
        state :math:`|\phi\rangle` (called target) and, possibly, by the operator
        :math:`U`. If :math:`U` is not specified, it is assumed :math:`|\Phi\rangle = |\phi\rangle`.

        The Monte Carlo estimator of I is:

        .. math::

            I = \mathbb{E}_{χ}[ I_{loc}(\sigma,\eta) ] =
                \mathbb{E}_{χ}\left[\frac{⟨\sigma|\Phi⟩ ⟨\eta|\Psi⟩}{⟨σ|\Psi⟩ ⟨η|\Phi⟩}\right]

        where the sampled probability distribution :math:`χ` is defined as:

        .. math::

            \chi(\sigma, \eta) = \frac{|\psi(\sigma)|^2 |\Phi(\eta)|^2}{
            \langle\Psi|\Psi\rangle  \langle\Phi|\Phi\rangle}.

        In practice, since I is a real quantity, :math:`\rm{Re}[I_{loc}(\sigma,\eta)]`
        is used. This estimator can be utilized both when :math:`|\Phi\rangle =|\phi\rangle` and
        when :math:`|\Phi\rangle =U|\phi\rangle`, with :math:`U` a (unitary or non-unitary) operator.
        In the second case, we have to sample from :math:`U|\phi\rangle` and this is implemented in
        the function :class:`netket_pro.infidelity.InfidelityUPsi` .

        This works only with the operators provdided in the package.
        We remark that sampling from :math:`U|\phi\rangle` requires to compute connected elements of
        :math:`U` and so is more expensive than sampling from an autonomous state.
        The choice of this estimator is specified by passing  :code:`sample_Uphi=True`,
        while the flag argument :code:`is_unitary` indicates whether :math:`U` is unitary or not.

        If :math:`U` is unitary, the following alternative estimator can be used:

        .. math::

            I = \mathbb{E}_{χ'}\left[ I_{loc}(\sigma, \eta) \right] =
                \mathbb{E}_{χ}\left[\frac{\langle\sigma|U|\phi\rangle \langle\eta|\psi\rangle}{
                \langle\sigma|U^{\dagger}|\psi\rangle ⟨\eta|\phi⟩} \right].

        where the sampled probability distribution :math:`\chi` is defined as:

        .. math::

            \chi'(\sigma, \eta) = \frac{|\psi(\sigma)|^2 |\phi(\eta)|^2}{
                \langle\Psi|\Psi\rangle  \langle\phi|\phi\rangle}.

        This estimator is more efficient since it does not require to sample from
        :math:`U|\phi\rangle`, but only from :math:`|\phi\rangle`.
        This choice of the estimator is the default and it works only
        with `is_unitary==True` (besides :code:`sample_Uphi=False` ).
        When :math:`|\Phi⟩ = |\phi⟩` the two estimators coincides.

        To reduce the variance of the estimator, the Control Variates (CV) method can be applied. This consists
        in modifying the estimator into:

        .. math::

            I_{loc}^{CV} = \rm{Re}\left[I_{loc}(\sigma, \eta)\right] - c \left(|1 - I_{loc}(\sigma, \eta)^2| - 1\right)

        where :math:`c ∈ \mathbb{R}`. The constant c is chosen to minimize the variance of
        :math:`I_{loc}^{CV}` as:

        .. math::

            c* = \frac{\rm{Cov}_{χ}\left[ |1-I_{loc}|^2, \rm{Re}\left[1-I_{loc}\right]\right]}{
                \rm{Var}_{χ}\left[ |1-I_{loc}|^2\right] },

        where :math:`\rm{Cov}_{χ}\left[\cdot, \cdot\right]` indicates the covariance and :math:`\rm{Var}\left[\cdot\right]`
        the variance of the given local estimators over the distribution :math:`χ.
        In the relevant limit :math:`|\Psi⟩ \rightarrow|\Phi⟩`, we have :math:`c^\star \rightarrow -1/2`. The value :math:`-1/2` is
        adopted as default value for c in the infidelity
        estimator. To not apply CV, set c=0.

        Args:
            target_state: target variational state :math:`|\phi⟩`.
            optimizer: the optimizer to use to use (from optax)
            variational_state: the variational state to train
            U: operator :math:`\hat{U}`.
            U_dagger: dagger operator :math:`\hat{U^\dagger}`.
            cv_coeff: Control Variates coefficient c.
            is_unitary: flag specifiying the unitarity of :math:`\hat{U}`. If True with
                :code:`sample_Uphi=False`, the second estimator is used.
            dtype: The dtype of the output of expectation value and gradient.
            sample_Uphi: flag specifiying whether to sample from :math:`|ϕ⟩` or from :math:`U|ϕ⟩` .
                If False with :code:`is_unitary=False`, an error occurs.
            preconditioner: Determines which preconditioner to use for the loss gradient.
                This must be a tuple of `(object, solver)` as documented in the section
                `preconditioners` in the documentation. The standard preconditioner
                included with NetKet is Stochastic Reconfiguration. By default, no
                preconditioner is used and the bare gradient is passed to the optimizer.
        """
        super().__init__(
            variational_state, optimizer, minimized_quantity_name="Infidelity"
        )

        if target_state is variational_state:
            raise ValueError(
                "Target state and variational_state must be two different objects."
            )

        self._cv = cv_coeff

        self._preconditioner = preconditioner

        self._I_op = InfidelityOperator(
            target_state,
            U=U,
            U_dagger=U_dagger,
            V=V,
            is_unitary=is_unitary,
            cv_coeff=cv_coeff,
            sample_Uphi=sample_Uphi,
            resample_fraction=resample_fraction,
        )

    def reset_step(self):
        """
        Resets the state of the driver at the beginning of a new step.

        This method is called at the beginning of every step in the optimization.
        """
        self.state.reset()
        self._I_op.target.reset()

    def compute_loss_and_update(self):
        infidelity, grad = self.state.expect_and_grad(self._I_op)

        dp = self.preconditioner(self.state, grad, self.step_count)

        return infidelity, dp

    def run(
        self,
        n_iter,
        out=None,
        *args,
        target_infidelity=None,
        callback: Callable[
            [int, dict, "AbstractVariationalDriver"], bool
        ] = lambda *x: True,
        **kwargs,
    ):
        """
        Executes the Infidelity optimisation, updating the weights of the network
        stored in this driver for `n_iter` steps and dumping values of the observables `obs`
        in the output `logger`. If no logger is specified, creates a json file at `out`,
        overwriting files with the same prefix.

        Args:
            n_iter: the total number of iterations
            out: A logger object, or an iterable of loggers, to be used to store simulation log and data.
            obs: An iterable containing all observables that should be computed
            target_infidelity: An optional floating point number that specifies when to stop the optimisation.
                This is used to construct a {class}`netket.callbacks.ConvergenceStopping` callback that stops
                the optimisation when that value is reached. You can also build that object manually for more
                control on the stopping criteria.
            step_size: Every how many steps should observables be logged to disk (default=1)
            show_progress: If true displays a progress bar (default=True)
            callback: Callable or list of callable callback functions to stop training given a condition
        """
        callbacks = to_iterable(callback)

        if target_infidelity is not None:
            cb = ConvergenceStopping(target_infidelity, smoothing_window=20, patience=5)
            callbacks = callbacks + (cb,)

        super().run(n_iter, out, *args, callback=callbacks, **kwargs)

    @property
    def cv(self) -> Optional[float]:
        """
        Return the coefficient for the Control Variates
        """
        return self._cv

    @property
    def infidelity(self) -> Stats:
        """
        Return MCMC statistics for the expectation value of observables in the
        current state of the driver.
        """
        return self._loss_stats

    @property
    def preconditioner(self):
        """
        The preconditioner used to modify the gradient.

        This is a function with the following signature

        .. code-block:: python

            precondtioner(vstate: VariationalState,
                          grad: PyTree,
                          step: Optional[Scalar] = None)

        Where the first argument is a variational state, the second argument
        is the PyTree of the gradient to precondition and the last optional
        argument is the step, used to change some parameters along the
        optimisation.

        Often, this is taken to be :func:`nk.optimizer.SR`. If it is set to
        `None`, then the identity is used.
        """
        return self._preconditioner

    @preconditioner.setter
    def preconditioner(self, val: Optional[PreconditionerT]):
        if val is None:
            val = identity_preconditioner

        self._preconditioner = val

    def __repr__(self):
        return (
            "InfidelityOptimizer("
            + f"\n  step_count = {self.step_count},"
            + f"\n  state = {self.state})"
        )
