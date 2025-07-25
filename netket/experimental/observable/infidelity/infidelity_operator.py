import jax.numpy as jnp
import flax
import jax

from netket.experimental.observable import AbstractObservable

from netket.utils.types import DType
from netket.utils.numbers import is_scalar
from netket.vqs import VariationalState, FullSumState, MCState
from netket.operator import AbstractOperator
from netket.jax import HashablePartial


class InfidelityOperator(AbstractObservable):
    r"""
    Infidelity operator computing the infidelity between an input variational state :math:`|\Psi\rangle` and a target state :math:`|\Phi\rangle`.

    The target state can be defined in two ways:

    1. as a variational state that is passed as `target_state`.
    2. as a state obtained by applying an operator :math:`U` to a variational state :math:`|\Phi\rangle`, i.e., :math:`|\Phi\rangle \equiv U|\Phi\rangle`.

    The infidelity :math:`I` among two variational states :math:`|\Psi\rangle` and :math:`|\Phi\rangle` is defined as:

    .. math::

        I = 1 - \frac{|\langle\Psi|\Phi\rangle|^2}{\langle\Psi|\Psi\rangle \langle\Phi|\Phi\rangle} = 1 - \frac{\langle\Psi|\hat{I}_{op}|\Psi\rangle}{\langle\Psi|\Psi\rangle},

    where:

    .. math::

        \hat{I}_{op} = \frac{|\Phi\rangle\langle\Phi|}{\langle\Phi|\Phi\rangle}.

    The Monte Carlo estimator of :math:`I` is:

    .. math::

        I = \mathbb{E}_{\chi}[ I_{loc}(x,y) ] = \mathbb{E}_{\chi}\left[ \frac{\langle x|\Phi\rangle \langle y|\Psi\rangle}{\langle x|\Psi\rangle \langle y|\Phi\rangle} \right],

    where :math:`\chi(x, y) = \frac{|\Psi(x)|^2 |\Phi(y)|^2}{\langle\Psi|\Psi\rangle \langle\Phi|\Phi\rangle}` is the joint Born distribution. This estimator
    can be utilized both when :math:`|\Phi\rangle = |\Phi\rangle` and when :math:`|\Phi\rangle = U|\Phi\rangle`, with :math:`U` a (unitary or
    non-unitary) operator. We remark that sampling from :math:`U|\Phi\rangle` is more expensive than sampling from an autonomous state.

    For details see `Sinibaldi et al. <https://quantum-journal.org/papers/q-2023-10-10-1131/>`_ and `Gravina et al. <https://quantum-journal.org/papers/q-2025-07-22-1803/>`_.
    """

    def __init__(
        self,
        target_state: VariationalState,
        *,
        operator: AbstractOperator = None,
        cv_coeff: float | None = -0.5,
        dtype: DType | None = None,
    ):
        """
        Args:
            target_state: The target state :math:`|\\Phi\rangle` against which to compute the infidelity.
                This can be any VariationalState (MCState, FullSumState, etc.).
            operator: Optional operator :math:`U` to be applied to the target state, such that :math:`|\\Phi\rangle \\equiv U |\\Phi\rangle`.
                If None, the target state is used directly. When provided, the infidelity is computed
                with respect to the transformed state :math:`U |\\Phi\rangle`.
            cv_coeff: Optional control variate coefficient for variance reduction in Monte Carlo
                estimation (see `Sinibaldi et al. <https://quantum-journal.org/papers/q-2023-10-10-1131/>`).
                If None, no control variate is used. Default to the optimal value -0.5.
            dtype: Data type for computations.
        """
        super().__init__(target_state.hilbert)

        if not isinstance(target_state, VariationalState):
            raise TypeError("The first argument should be a variational target.")

        if cv_coeff is not None:
            cv_coeff = jnp.array(cv_coeff)

            if (not is_scalar(cv_coeff)) or jnp.iscomplex(cv_coeff):
                raise TypeError("`cv_coeff` should be a real scalar number or None.")

            if isinstance(target_state, FullSumState):
                cv_coeff = None

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

            logpsi_fun = HashablePartial(_logpsi_fun, target_state._apply_fun)

            if isinstance(target_state, MCState):
                self._target_state = MCState(
                    sampler=target_state.sampler,
                    apply_fun=logpsi_fun,
                    n_samples=target_state.n_samples,
                    variables=flax.core.copy(
                        target_state.variables, {"operator": operator}
                    ),
                )
            if isinstance(target_state, FullSumState):
                self._target_state = FullSumState(
                    hilbert=target_state.hilbert,
                    apply_fun=logpsi_fun,
                    variables=flax.core.copy(
                        target_state.variables, {"operator": operator}
                    ),
                )
        else:
            self._target_state = target_state

        self._cv_coeff = cv_coeff
        self._dtype = dtype

    @property
    def target_state(self):
        return self._target_state

    @property
    def cv_coeff(self):
        return self._cv_coeff

    def __repr__(self):
        return f"InfidelityOperator(target_state={self.target_state}, cv_coeff={self.cv_coeff})"
