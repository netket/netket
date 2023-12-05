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

import abc
from typing import Optional

import numpy as np

import jax
import jax.numpy as jnp
from jax.nn.initializers import normal

from flax import core as fcore
from flax.core.scope import CollectionFilter, DenyList  # noqa: F401

import netket.jax as nkjax
from netket.operator import AbstractOperator
from netket.hilbert import AbstractHilbert
from netket.stats import Stats
from netket.utils.types import PyTree, PRNGKeyT, NNInitFunc
from netket.utils.dispatch import dispatch
from netket.utils.optional_deps import import_optional_dependency


class VariationalState(abc.ABC):
    """Abstract class for variational states representing either pure states
    or mixed quantum states.

    A variational state is a quantum state depending on a set of
    parameters, and that supports operations such
    as computing quantum expectation values and their gradients.

    A Variational stat can be serialized using flax's msgpack machinery.
    See `their docs <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_.

    """

    def __init__(self, hilbert: AbstractHilbert):
        """Initialize the Abstract base class of a Variational State defined
        on an hilbert space.

        Args:
            hilbert: The hilbert space upon which this state is defined.

        """
        self._hilbert = hilbert  # type: AbstractHilbert

        self._model_state = {}  # type: PyTree
        self._parameters = {}  # type: PyTree

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The descriptor of the Hilbert space
        on which this variational state is defined.
        """
        return self._hilbert

    @property
    def parameters(self) -> PyTree:
        r"""The pytree of the parameters of the model."""
        return fcore.copy(self._parameters, {})

    @property
    def n_parameters(self) -> int:
        r"""The total number of parameters in the model."""
        return nkjax.tree_size(self.parameters)

    @parameters.setter
    def parameters(self, pars: PyTree):
        self._parameters = pars
        self.reset()

    @property
    def model_state(self) -> Optional[PyTree]:
        r"""The optional pytree with the mutable state of the model."""
        return fcore.copy(self._model_state, {})

    @model_state.setter
    def model_state(self, state: PyTree):
        self._model_state = state
        self.reset()

    @property
    def variables(self) -> PyTree:
        r"""The PyTree containing the parameters and state of the model,
        used when evaluating it.
        """
        return {"params": self.parameters, **self.model_state}

    @variables.setter
    def variables(self, var: PyTree):
        self.model_state, self.parameters = fcore.pop(var, "params")

    def init_parameters(
        self, init_fun: Optional[NNInitFunc] = None, *, seed: Optional[PRNGKeyT] = None
    ):
        r"""
        Re-initializes all the parameters with the provided initialization function,
        defaulting to the normal distribution of standard deviation 0.01.

        .. warning::

            The init function will not change the dtype of the parameters, which is
            determined by the model. DO NOT SPECIFY IT INSIDE THE INIT FUNCTION

        Args:
            init_fun: a jax initializer such as :func:`jax.nn.initializers.normal`.
                Must be a Callable taking 3 inputs, the jax PRNG key, the shape and the
                dtype, and outputting an array with the valid dtype and shape. If left
                unspecified, defaults to :code:`jax.nn.initializers.normal(stddev=0.01)`
            seed: Optional seed to be used. The seed is synced across all MPI processes.
                If unspecified, uses a random seed.
        """
        if init_fun is None:
            init_fun = normal(stddev=0.01)

        rng = nkjax.PRNGSeq(nkjax.PRNGKey(seed))

        def new_pars(par):
            return jnp.asarray(
                init_fun(rng.take(1)[0], shape=par.shape, dtype=par.dtype),
                dtype=par.dtype,
            )

        self.parameters = jax.tree_map(new_pars, self.parameters)

    def reset(self):
        r"""Resets the internal cache of th variational state.
        Called automatically when the parameters/state is updated.
        """

    def expect(self, O: AbstractOperator) -> Stats:
        r"""Estimates the quantum expectation value for a given operator
        :math:`O` or generic observable.
        In the case of a pure state :math:`\psi` and an operator, this is
        :math:`\langle O\rangle= \langle \Psi|O|\Psi\rangle/\langle\Psi|\Psi\rangle`
        otherwise for a  mixed state :math:`\rho`, this is
        :math:`\langle O\rangle= \textrm{Tr}[\rho \hat{O}]/\textrm{Tr}[\rho]`.

        Args:
            O: the operator or observable for which to compute the expectation
                value.

        Returns:
            An estimation of the quantum expectation value
            :math:`\langle O\rangle`.
        """
        return expect(self, O)

    def grad(
        self,
        Ô,
        *,
        use_covariance: Optional[bool] = None,
        mutable: Optional[CollectionFilter] = None,
    ) -> PyTree:
        r"""Estimates the gradient of the quantum expectation value of a given operator O.

        Args:
            op (netket.operator.AbstractOperator): the operator O.
            is_hermitian: optional override for whether to use or not the hermitian logic. By default
                it's automatically detected.

        Returns:
            array: An estimation of the average gradient of the quantum expectation value <O>.
        """
        r = self.expect_and_grad(Ô, use_covariance=use_covariance, mutable=mutable)
        return r[1]

    def expect_and_grad(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
        **kwargs,
    ) -> tuple[Stats, PyTree]:
        r"""Estimates the quantum expectation value and its gradient
        for a given operator :math:`O`.

        Args:
            O: The operator :math:`O` for which expectation value and
                gradient are computed.
            mutable: Can be bool, str, or list. Specifies which collections in the
                     model_state should be treated as  mutable: bool: all/no collections
                     are mutable. str: The name of a single mutable  collection. list: A
                     list of names of mutable collections. This is used to mutate the state
                     of the model while you train it (for example to implement BatchNorm. Consult
                     `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                     for a more in-depth explanation).
            use_covariance: whether to use the covariance formula, usually reserved for
                hermitian operators,
                :math:`\textrm{Cov}[\partial\log\psi, O_{\textrm{loc}}\rangle]`

        Returns:
            An estimate of the quantum expectation value <O>.
            An estimate of the gradient of the quantum expectation value <O>.
        """
        if mutable is None:
            mutable = self.mutable

        return expect_and_grad(self, O, mutable=mutable, **kwargs)

    def expect_and_forces(
        self,
        O: AbstractOperator,
        *,
        mutable: Optional[CollectionFilter] = None,
    ) -> tuple[Stats, PyTree]:
        r"""Estimates the quantum expectation value and the corresponding force
        vector for a given operator O.

        The force vector :math:`F_j` is defined as the covariance of log-derivative
        of the trial wave function and the local estimators of the operator. For complex
        holomorphic states, this is equivalent to the expectation gradient
        :math:`\frac{\partial\langle O\rangle}{\partial(\theta_j)^\star} = F_j`. For real-parameter states,
        the gradient is given by :math:`\frac{\partial\partial_j\langle O\rangle}{\partial\partial_j\theta_j} = 2 \textrm{Re}[F_j]`.

        Args:
            O: The operator O for which expectation value and force are computed.
            mutable: Can be bool, str, or list. Specifies which collections in
                the model_state should be treated as  mutable: bool: all/no
                collections are mutable. str: The name of a single mutable
                collection. list: A list of names of mutable collections. This is
                used to mutate the state of the model while you train it (for
                example to implement BatchNorm. Consult
                `Flax's Module.apply documentation <https://flax.readthedocs.io/en/latest/_modules/flax/linen/module.html#Module.apply>`_
                for a more in-depth explanation).

        Returns:
            An estimate of the quantum expectation value <O>.
            An estimate of the force vector
            :math:`F_j = \textrm{Cov}[\partial_j\log\psi, O_{\textrm{loc}}]`.
        """
        if mutable is None:
            mutable = self.mutable

        return expect_and_forces(self, O, mutable=mutable)

    # @abc.abstractmethod
    def quantum_geometric_tensor(self, qgt_type):
        r"""Computes an estimate of the quantum geometric tensor G_ij.

        This function returns a linear operator that can be used to apply G_ij to a
        given vector or can be converted to a full matrix.

        Args:
            qgt_type: the optional type of the quantum geometric tensor. By default it
                is automatically selected.

        Returns:
            nk.optimizer.LinearOperator: A linear operator representing the quantum
                geometric tensor.
        """
        raise NotImplementedError  # pragma: no cover

    def to_array(self, normalize: bool = True) -> jnp.ndarray:
        """
        Returns the dense-vector representation of this state.

        Args:
            normalize: If True, the vector is normalized to have L2-norm 1.

        Returns:
            An exponentially large vector representing the state in the computational
            basis.
        """
        return NotImplemented  # pragma: no cover

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert the variational state to a qutip's ket Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        q_dims = [list(self.hilbert.shape), [1 for i in range(self.hilbert.size)]]
        return qutip.Qobj(np.asarray(self.to_array()), dims=q_dims)


class VariationalMixedState(VariationalState):
    __module__ = "netket.vqs"

    def __init__(self, hilbert, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hilbert_physical = hilbert

    @property
    def hilbert_physical(self) -> AbstractHilbert:
        return self._hilbert_physical

    def to_matrix(self, normalize: bool = True) -> jnp.ndarray:
        """
        Returns the dense-matrix representation of this operator.

        Args:
            normalize: If True, the matrix is normalized to have trace 1.

        Returns:
            An exponentially large matrix representing the state in the computational
            basis.
        """
        return NotImplemented  # pragma: no cover

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert this mixed state to a qutip density matrix Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        q_dims = [list(self.hilbert_physical.shape), list(self.hilbert_physical.shape)]
        return qutip.Qobj(np.asarray(self.to_matrix()), dims=q_dims)


@dispatch.abstract
def expect(vstate: VariationalState, operator: AbstractOperator):
    """
    Computes the expectation value of the given operator over the
    variational state.

    Additional Information:
        To implement `vstate.expect` for a custom operator, implement
        the multiple-dispatch (plum-dispatc) based method according

        .. code:

            @nk.vqs.expect.register
            expect(vstate : VStateType operator: OperatorType):
                return ...

    Args:
        vstate: The VariationalState
        operator: The Operator or SuperOperator.

    Returns:
        The expectation value wrapped in a `Stats` object.
    """


# default dispatch where use_covariance is not specified
# Give it an higher precedence so this is always executed first, no matter what, if there
# is a dispatch ambiguity.
# This is not needed, but makes the dispatch logic work fine even if the users write weak
# signatures (eg: if an users defines `expect_grad(vs: MCState, op: MyOperator, use_cov: Any)`
# instead of `expect_grad(vs: MCState, op: MyOperator, use_cov: bool)`
# there would be a resolution error because the signature defined by the user is stricter
# for some arguments, but the one below here is stricter for `use_covariance` which is
# set to bool. Since this signature below, in the worst case, does nothing, this ensures
# that `expect_and_grad` is more user-friendly.
@dispatch.abstract
def expect_and_grad(
    vstate: VariationalState,
    operator: AbstractOperator,
    *args,
    mutable: CollectionFilter,
    **kwargs,
):
    r"""Estimates the quantum expectation value and its gradient for a given operator O.

    See `VariationalState.expect_and_grad` docstring for more information.

    Additional Information:
        To implement `vstate.expect` for a custom operator, implement
        the multiple-dispatch (plum-dispatc) based method according to the signature below.

        .. code:

            @nk.vqs.expect_and_grad.register
            expect_and_grad(vstate : VStateType, operator: OperatorType,
                            use_covariance : bool/Literal[True]/Literal[False], * mutable)
                return ...
    """


@dispatch.abstract
def expect_and_forces(
    vstate: VariationalState,
    operator: AbstractOperator,
    *args,
    mutable: CollectionFilter,
    **kwargs,
):
    r"""Estimates the quantum expectation value and corresponding force vector for a given operator O.

    See `VariationalState.expect_and_forces` docstring for more information.

    Additional Information:
        To implement `vstate.expect` for a custom operator, implement
        the multiple-dispatch (plum-dispatc) based method according to the signature below.

        .. code:

            @nk.vqs.expect_and_forces.register
            expect_and_forces(vstate : VStateType, operator: OperatorType,
                              use_covariance : bool/Literal[True]/Literal[False], * mutable)
                return ...
    """
