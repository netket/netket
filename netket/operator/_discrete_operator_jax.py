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
from typing import Tuple

import numpy as np
import jax.numpy as jnp

from jax.experimental.sparse import JAXSparse, BCOO

from netket.operator import DiscreteOperator


class DiscreteJaxOperator(DiscreteOperator):
    r"""This class should be inherited by DiscreteOperators which
    wish to declare jax-compatibility.

    `DiscreteJaxOperator`s should be declared following a scheme like

    Examples:

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
        raise NotImplementedError

    @abc.abstractmethod
    def get_conn_padded(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        r"""Finds the connected elements of the Operator.

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
    ) -> Tuple[np.ndarray, np.ndarray]:
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
        x = self.hilbert.all_states()
        n = x.shape[0]
        xp, mels = self.get_conn_padded(x)
        a = mels.ravel()
        i = np.broadcast_to(np.arange(n)[..., None], mels.shape).ravel()
        j = self.hilbert.states_to_numbers(xp).ravel()
        ij = np.concatenate((i[:, None], j[:, None]), axis=1)
        return BCOO((a, ij), shape=(n, n))

    def to_dense(self) -> np.ndarray:
        return self.to_sparse().todense()
