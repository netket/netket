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

import jax
import jax.numpy as jnp
from jax.experimental.sparse import JAXSparse, BCOO, BCSR

from netket.operator import AbstractOperator, DiscreteOperator
from netket.utils.optional_deps import import_optional_dependency


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

    Most operators can be converted to
    their jax-enabled counterparts by calling the method
    :meth:`~netket.operator.DiscreteJaxOperator.to_jax_operator`.
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
        self,
        x: np.ndarray,
        sections: np.ndarray,
        pad: bool = False,
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
        xp = np.array(xp)
        mels = np.array(mels)

        if pad:
            n_conns = mels.shape[-1]
            sections[:] = np.arange(1, len(x) + 1) * n_conns
        else:
            n_conns = np.count_nonzero(mels, axis=-1)
            sections[:] = np.cumsum(n_conns)

        xp = xp.reshape(-1, xp.shape[-1])
        mels = mels.reshape(-1)

        if not pad:
            mask = np.where(mels != 0)
            xp = xp[mask]
            mels = mels[mask]

        return xp, mels

    @jax.jit
    def n_conn(self, x, out=None) -> jax.Array:
        r"""Return the number of (non-zero) connected entries to `x`.

        .. warning::

            This is not the True number of connected entries, because some elements
            might appear twice (however this should not be too common.)

            Note that this deviates from the Numba implementation, and can generally
            return a smaller number of connected entries.

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            out (array): If None an output array is allocated.

        Returns:
            array: The number of connected states x' for each x[i].
        """

        _, mels = self.get_conn_padded(x)
        nonzeros = jnp.abs(x) > 0
        _n_conn = nonzeros.sum(axis=-1)

        if out is None:
            out = _n_conn
        else:
            raise ValueError("The out argument is not supported for jax operators.")
            # cannot do this inside of jit
            # out[:] = _n_conn
        return out

    def to_sparse(self, jax_: bool = False) -> JAXSparse:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Args:
            jax_: If True, returns an experimental Jax sparse matrix. If False,
                returns a normal scipy CSR matrix. False by default.

        Returns:
            The sparse jax matrix representation of the operator.
        """

        if not jax_:
            # calls the get_conn_flattened code path
            return super().to_sparse()

        x = self.hilbert.all_states()
        n = x.shape[0]
        xp, mels = self.get_conn_padded(x)
        ip = self.hilbert.states_to_numbers(xp)
        # sum duplicates and remove zeros in every row
        # this also sorts the indices
        A = BCOO((mels, ip[:, :, None]), shape=(n, n)).sum_duplicates()
        # remove batching and turn it into a normal COO matrix
        A = A.update_layout(n_batch=0)
        # turn it into BCSR
        A = BCSR.from_bcoo(A)
        return A

    def to_dense(self) -> jax.Array:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a jax Array.
        """
        return self.to_sparse(jax_=True).todense()

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert the operator to a qutip's Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        sparse_mat_scipy = self.to_sparse(jax_=False)

        return qutip.Qobj(
            sparse_mat_scipy, dims=[list(self.hilbert.shape), list(self.hilbert.shape)]
        )

    def __matmul__(self, other):
        if isinstance(other, JAXSparse):
            return self.apply(other.todense())
        elif isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
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

    def to_jax_operator(self) -> "DiscreteJaxOperator":
        """
        Return the JAX version of this operator.

        If this is a JAX operator does nothing.
        """
        return self
