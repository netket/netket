import abc

from functools import partial
from typing import Any, Optional, Tuple

import flax
from flax.core.frozen_dict import FrozenDict

from netket.operator import AbstractOperator, LocalLiouvillian
from netket.hilbert import AbstractHilbert, DoubledHilbert
from netket.stats import Stats

PyTree = Any


class VariationalState(abc.ABC):
    """Abstract class for variational states representing either pure states
    or mixed quantum states.

    A variational state is a quantum state depending on a set of
    parameters, and that supports operations such
    as computing quantum expectation values and their gradients.

    A Variational stat can be serialized using flax's msgpack machinery.
    See `their docs <https://flax.readthedocs.io/en/latest/flax.serialization.html>`_.

    """

    _hilbert: AbstractHilbert
    """The hilbert space on which this state is defined."""

    parameters: PyTree
    """The PyTree of variational parameters"""

    def __init__(self, hilbert):
        self._hilbert = hilbert

        self._model_state = {}
        self._parameters = {}

    @property
    def hilbert(self) -> AbstractHilbert:
        r"""The descriptor of the Hilbert space
        on which this variational state is defined.
        """
        return self._hilbert

    @property
    def parameters(self) -> PyTree:
        r"""The pytree of the parameters of the model."""
        return self._parameters

    @parameters.setter
    def parameters(self, pars: PyTree):
        self._parameters = pars

    @property
    def model_state(self) -> Optional[PyTree]:
        r"""The optional pytree with the mutable state of the model."""
        return self._model_state

    @model_state.setter
    def model_state(self, state: PyTree):
        self._model_state = state

    @property
    def variables(self) -> PyTree:
        r"""The PyTreee containing the paramters and state of the model,
        used when evaluating it.
        """
        return flax.core.freeze({"params": self.parameters, **self.model_state})

    @variables.setter
    def variables(self, vars) -> PyTree:
        if not isinstance(vars, FrozenDict):
            vars = flax.core.freeze(vars)

        self.model_state, self.parameters = vars.pop("params")

    def reset(self):
        pass

    @abc.abstractmethod
    def expect(self, Ô: AbstractOperator) -> Stats:
        r"""Estimates the quantum expectation value for a given operator O.
            In the case of a pure state $\psi$, this is $<O>= <Psi|O|Psi>/<Psi|Psi>$
            otherwise for a mixed state $\rho$, this is $<O> = \Tr[\rho \hat{O}/\Tr[\rho]$.

        Args:
            Ô (netket.operator.AbstractOperator): the operator O.

        Returns:
            An estimation of the quantum expectation value <O>.
        """
        raise NotImplementedError

    def grad(self, Ô) -> PyTree:
        r"""Estimates the gradient of the quantum expectation value of a given operator O.

        Args:
            op (netket.operator.AbstractOperator): the operator O.

        Returns:
            array: An estimation of the average gradient of the quantum expectation value <O>.
        """
        return self.expect_and_grad(Ô)[1]

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        mutable=None,
    ) -> Tuple[Stats, PyTree]:
        r"""Estimates both the gradient of the quantum expectation value of a given operator O.

        Args:
            Ô (netket.operator.AbstractOperator): the operator O

        Returns:
            An estimation of the quantum expectation value <O>.
            An estimation of the average gradient of the quantum expectation value <O>.
        """
        raise NotImplementedError

    # @abc.abstractmethod
    def quantum_geometric_tensor(self, sr):
        r"""Computes an estimate of the quantum geometric tensor G_ij.

        This function returns a linear operator that can be used to apply G_ij to a given vector
        or can be converted to a full matrix.

        Returns:
           A linear operator representing the quantum geometric tensor.
        """
        raise NotImplementedError


class VariationalMixedState(VariationalState):
    def __init__(self, hilbert, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hilbert_physical = hilbert

    @property
    def hilbert_physical(self) -> AbstractHilbert:
        return self._hilbert_physical

    def expect(self, Ô: AbstractOperator) -> Stats:
        # If it is super-operator treat, they act on the same space so
        # the expectation value is standard.
        if self.hilbert == Ô.hilbert:
            return super().expect(Ô)
        elif self.hilbert_physical == Ô.hilbert:
            return self.expect_operator(Ô)
        else:
            return NotImplemented

    def expect_and_grad(
        self,
        Ô: AbstractOperator,
        mutable: bool = None,
    ) -> Tuple[Stats, PyTree]:
        # do the computation in super-operator space
        if self.hilbert == Ô.hilbert:
            return super().expect_and_grad(Ô, mutable=mutable)
        elif self.hilbert_physical == Ô.hilbert:
            return super().expect_and_grad(Ô, mutable=mutable)
        else:
            return NotImplemented

    @abc.abstractmethod
    def expect_operator(self, Ô: AbstractOperator) -> Stats:
        raise NotImplementedError

    def grad_operator(self, Ô: AbstractOperator) -> Stats:
        return self.expect_and_grad_operator(Ô)[1]

    # @abc.abstractmethod
    def expect_and_grad_operator(self, Ô: AbstractOperator) -> Stats:
        raise NotImplementedError
