# Copyright 2021-2023 The NetKet Authors - All rights reserved.
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

import numpy as np
import jax.numpy as jnp
from jax.experimental.sparse import JAXSparse, BCOO

from netket.operator import AbstractOperator, DiscreteOperator


class DiscreteJaxOperator(DiscreteOperator):
    r"""Abstract base class for discrete operators that can
    be manipulated inside of jax function transformations.

    Any operator inheriting from this base class follows the
    :class:`netket.operator.DiscreteOperator` interface but
    can additionally be used inside of :func:`jax.jit`,
    :func:`jax.grad`, :func:`jax.vmap` or similar transformations.
    When passed to those functions, jax-compatible operators
    must not be passed as static arguments but as standard
    arguments, and they will not trigger recompilation if
    only the coefficients have changed.

    Some operators, such as :class:`netket.operator.Ising` or
    :class:`netket.operator.PauliStrings` can be converted to
    their jax-enabled counterparts by calling the method
    :meth:`~netket.operator.PauliStrings.to_jax_operator`.
    Not all operators support this conversion, but as
    :class:`netket.operator.PauliStrings` are flexible, if you
    can convert or write your hamiltonian as a sum of pauli
    strings you will be able to use
    :class:`netket.operator.PauliStringsJax`.


    .. note::

        Jax does not support dynamically varying shapes, so not
        all operators can be written as jax operators, and even
        if they could be written as such, they might generate
        more connected elements than their Numba counterpart.


    .. note::

        :class:`netket.operator.DiscreteJaxOperator` require a
        particular version of the hamiltonian sampling rule,
        :func:`netket.sampler.rules.HamiltonianRuleJax`, that is
        compatible with Jax.


    Defining custom discrete operators that are Jax-compatible
    ----------------------------------------------------------

    This class should be inherited by DiscreteOperators which
    wish to declare jax-compatibility.

    Classes inheriting from `DiscreteJaxOperator`` should be
    declared following a scheme like the following. Do notice
    in particular the declaration of the pytree flattening and
    unflattening, following the standard APIs of Jax discussed
    in the `Jax Pytree documentation <https://jax.readthedocs.io/en/latest/pytrees.html#custom-pytrees-and-initialization>`_.

    .. code-block:: python

         from jax.tree_util import register_pytree_node_class

         @register_pytree_node_class
         class MyJaxOperator(DiscreteJaxOperator):
             def __init__(hilbert, ...):
                 super().__init__(hilbert)

             def tree_flatten(self):
                 array_data = ( ... ) # all arrays
                 struct_data = {'hilbert': self.hilbert,
                                 ... # all constant data
                                 }
                 return array_data, struct_data

             @classmethod
             def tree_unflatten(cls, struct_data, array_data):
                 ...
                 return cls(array_data['hilbert'], ...)

             @property
             def max_conn_size(self) -> int:
                 return ...

             def get_conn_padded(self, x):
                 ...
                 return xp, mels


    """

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        raise NotImplementedError  # pragma: no cover

    @abc.abstractmethod
    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator. This method
        can be executed inside of a Jax function transformation.

        Starting from a batch of quantum numbers :math:`x={x_1, ... x_n}` of
        size :math:`B \times M` where :math:`B` size of the batch and :math:`M`
        size of the hilbert space, finds all states :math:`y_i^1, ..., y_i^K`
        connected to every :math:`x_i`.

        Returns a matrix of size :math:`B \times K_{max} \times M` where
        :math:`K_{max}` is the maximum number of connections for every
        :math:`y_i`.

        Args:
            x : A N-tensor of shape :math:`(...,hilbert.size)` containing
                the batch/batches of quantum numbers :math:`x`.

        Returns:
            **(x_primes, mels)**: The connected states x', in a N+1-tensor and an
            N-tensor containing the matrix elements :math:`O(x,x')`
            associated to each x' for every batch.
        """

    def get_conn_flattened(
        self, x: np.ndarray, sections: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

        Starting from a given quantum number :math:`x`, it finds all
        other quantum numbers  :math:`x'` such that the matrix element
        :math:`O(x,x')` is different from zero. In general there will be
        several different connected states :math:`x'` satisfying this
        condition, and they are denoted here :math:`x'(k)`, for
        :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape
        :code:`(batch_size,hilbert.size)`.

        Args:
            x: A matrix of shape `(batch_size, hilbert.size)`
                containing the batch of quantum numbers x.
            sections: An array of sections for the flattened x'.
                See numpy.split for the meaning of sections.

        Returns:
            (matrix, array): The connected states x', flattened together in
                a single matrix.
                An array containing the matrix elements :math:`O(x,x')`
                associated to each x'.

        """
        xp, mels = self.get_conn_padded(x)
        n_conns = mels.shape[1]
        xp = xp.reshape(-1, xp.shape[-1])
        mels = mels.reshape(-1)
        sections[:] = n_conns
        return xp, mels

    def n_conn(self, x, out=None) -> np.ndarray:
        r"""Return the number of states connected to x.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = jnp.full(x.shape[0], self.max_conn_size, dtype=np.int32)
        else:
            out[:] = self.max_conn_size
        return out

    def to_sparse(self) -> JAXSparse:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The sparse jax matrix representation of the operator.
        """
        x = self.hilbert.all_states()
        n = x.shape[0]
        xp, mels = self.get_conn_padded(x)
        a = mels.ravel()
        i = np.broadcast_to(np.arange(n)[..., None], mels.shape).ravel()
        j = self.hilbert.states_to_numbers(xp).ravel()
        ij = np.concatenate((i[:, None], j[:, None]), axis=1)
        return BCOO((a, ij), shape=(n, n))

    def to_dense(self) -> np.ndarray:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a jax Array.
        """
        return self.to_sparse().todense()

    def __matmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or isinstance(other, JAXSparse)
        ):
            return self.apply(other)
        elif isinstance(other, AbstractOperator):
            return self._op__matmul__(other)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or isinstance(other, JAXSparse)
        ):
            return NotImplemented
        elif isinstance(other, AbstractOperator):
            return self._op__rmatmul__(other)
        else:
            return NotImplemented
