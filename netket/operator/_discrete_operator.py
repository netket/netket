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

import numpy as np
import jax.numpy as jnp

from numba import jit
from scipy.sparse import csr_matrix as _csr_matrix
from scipy.sparse import issparse

from netket import config
from netket.hilbert import DiscreteHilbert
from netket.operator import AbstractOperator
from netket.utils.optional_deps import import_optional_dependency


class DiscreteOperator(AbstractOperator):
    r"""This class is the base class for operators defined on a
    discrete Hilbert space. Users interested in implementing new
    quantum Operators for discrete Hilbert spaces should derive
    their own class from this class
    """

    def __init__(self, hilbert: DiscreteHilbert):
        if not isinstance(hilbert, DiscreteHilbert):
            raise ValueError(
                "A Discrete Operator can only act upon a discrete Hilbert space."
            )
        super().__init__(hilbert)

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        raise NotImplementedError

    def get_conn_padded(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        if config.netket_experimental_sharding:
            raise RuntimeError(
                "When using Sharding mode, only jax operators are supported."
            )

        n_visible = x.shape[-1]
        n_samples = x.size // n_visible

        sections = np.empty(n_samples, dtype=np.int32)
        x_primes, mels = self.get_conn_flattened(
            x.reshape(-1, x.shape[-1]), sections, pad=True
        )

        n_primes = sections[0]

        x_primes_r = x_primes.reshape(*x.shape[:-1], n_primes, n_visible)
        mels_r = mels.reshape(*x.shape[:-1], n_primes)

        return x_primes_r, mels_r

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
        raise NotImplementedError(
            f"""
            The method get_conn_flattened has not been implemented for the object of
            type {type(self)}.

            This may happen if you defined a custom class inheriting from DiscreteOperator
            and you have not implemented this method. In that case, you should define
            `get_conn_flattened(self, x: array, sections: array)` according to the
            docstring provided on the documentation.

            Otherwise, please open an issue on netket's github repository.
            """
        )

    def get_conn(self, x: np.ndarray):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        Args:
            x: An array of shape `(hilbert.size, )` containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        Raise:
            ValueError: If the given quantum number is not compatible with the hilbert space.
        """
        if x.ndim != 1:
            raise ValueError(
                "get_conn does not support batches. Please use get_conn_flattened instead."
            )
        if x.shape[0] != self.hilbert.size:
            raise ValueError(
                "The given quantum numbers do not match the hilbert space because"
                f"it has shape {x.shape} of which[0] but expected {self.hilbert.size}."
            )

        return self.get_conn_flattened(
            x.reshape((1, -1)),
            np.ones(1),
        )

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
            out = np.empty(x.shape[0], dtype=np.int32)
        self.get_conn_flattened(x, out)
        out = self._n_conn_from_sections(out)

        return out

    @staticmethod
    @jit(nopython=True)
    def _n_conn_from_sections(out):
        low = 0
        for i in range(out.shape[0]):
            old_out = out[i]
            out[i] = out[i] - low
            low = old_out

        return out

    def to_sparse(self) -> _csr_matrix:
        r"""Returns the sparse matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The sparse matrix representation of the operator.
        """
        concrete_op = self.collect()
        hilb = self.hilbert

        x = hilb.all_states()

        sections = np.empty(x.shape[0], dtype=np.int32)
        x_prime, mels = concrete_op.get_conn_flattened(x, sections)

        sections1 = np.empty(sections.size + 1, dtype=np.int32)
        sections1[1:] = sections
        sections1[0] = 0
        # numbers = hilb.states_to_numbers(x_prime)
        try:
            # this is the original code, if inbetween states leave a constraint hilbert space,
            # (due to expansion of the operator for example) the except part tries to remove the states,
            # which are not in the constraint hilbert space. filter_jit is necessary to allow try...
            import equinox as eqx
            import jax
            numbers = eqx.filter_jit(hilb.states_to_numbers)(x_prime)
        except:
            # This part removes x_primes, which have mels sum 0
            # This is usefull, if inbetween states are not in the constraint hilbert space, as in this case
            # states_to_numbers will fail
            import jax
            jax.debug.print("starting")

            # test with hashes, but section 1 has to be changed as well
            x_prime_dict = {}
            for i in range(x_prime.shape[0]):
                key = tuple(x_prime[i])
                if key in x_prime_dict:
                    x_prime_dict[key] += mels[i]
                else:
                    x_prime_dict[key] = mels[i]

            # x_prime = np.array(list(x_prime_dict.keys()))
            # mels = np.array(list(x_prime_dict.values()))
            # non_zero_indices = np.nonzero(mels)[0]
            # x_prime = x_prime[non_zero_indices]
            # mels = mels[non_zero_indices]

            # x_prime = np.array(list(x_prime_dict.keys()))
            # mels = np.array(list(x_prime_dict.values()))

            x_prime_tmp, mels_tmp = zip(*x_prime_dict.items())
            x_prime_tmp = np.array(x_prime_tmp)
            mels_tmp = np.array(mels_tmp)

            # zero_indices = np.where(mels_tmp == 0)[0]
            # x_primes_to_remove = x_prime_tmp[zero_indices]
            
            jax.debug.print("removing prepared")
            mels_test = np.array([x_prime_dict[tuple(x_prime[i])] for i in range(x_prime.shape[0])])

            # unique_x_prime, indices = np.unique(x_prime, axis=0, return_index=True)
            # remove_mel = np.ones(unique_x_prime.shape[0], dtype=bool)
            # @jit(nopython=True)
            # def prepare(unique_x_prime, x_prime, mels, sections1, remove_mel):
            #     for i in range(unique_x_prime.shape[0]):
            #         for j in range(len(sections1)-1):
            #             sm = 0
            #             for k in range(sections1[j], sections1[j+1]):
            #                 if (x_prime[k] == unique_x_prime[i]).all():
            #                     sm += mels[k]
            #             if sm != 0:
            #                 remove_mel[i]=False
            #                 break
            #     return remove_mel
            # remove_mel = prepare(unique_x_prime, x_prime, mels, sections1, remove_mel)
            # x_primes_to_remove = unique_x_prime[remove_mel]



            # Create a mask for elements to keep (mels_test != 0)
            keep_mask = mels_test != 0

            # Apply the mask to all arrays at once
            x_prime = x_prime[keep_mask]
            mels = mels[keep_mask]
            # mels_test = mels_test[keep_mask]

            position = 0
            pindex = 0
            # removeset = set(map(hash, map(tuple, x_primes_to_remove)))
            for pindex in range(keep_mask.size):
                if ~keep_mask[pindex]:
                    # x_prime = np.delete(x_prime, position, axis=0)
                    # mels = np.delete(mels, position)
                    # mels_test = np.delete(mels_test, position)
                    sections1[sections1 > position] -= 1    
                else:
                    position += 1
                # pindex += 1

                if position % 10000 == 0:
                    jax.debug.print(f"position: {position} / {x_prime.shape[0]}")

            # position = x_prime.shape[0] - 1
            # while position >= 0:
            #     if mels[position] == 0:
            #         x_prime = np.delete(x_prime, position, axis=0)
            #         mels = np.delete(mels, position)
            #         sections1[sections1 > position] -= 1
            #     position -= 1

            jax.debug.print("removed")

            numbers = hilb.states_to_numbers(x_prime)

        # eliminate duplicates from numbers
        # rows_indices = compute_row_indices(hilb.states_to_numbers(x), sections1)

        return _csr_matrix(
            (mels, numbers, sections1),
            shape=(self.hilbert.n_states, self.hilbert.n_states),
        )

        # return _csr_matrix(
        #    (mels, (rows_indices, numbers)),
        #    shape=(self.hilbert.n_states, self.hilbert.n_states),
        # )


    def to_dense(self) -> np.ndarray:
        r"""Returns the dense matrix representation of the operator. Note that,
        in general, the size of the matrix is exponential in the number of quantum
        numbers, and this operation should thus only be performed for
        low-dimensional Hilbert spaces or sufficiently sparse operators.

        This method requires an indexable Hilbert space.

        Returns:
            The dense matrix representation of the operator as a Numpy array.
        """
        return self.to_sparse().todense().A

    def to_qobj(self):  # -> "qutip.Qobj"
        r"""Convert the operator to a qutip's Qobj.

        Returns:
            A :class:`qutip.Qobj` object.
        """
        qutip = import_optional_dependency("qutip", descr="to_qobj")

        return qutip.Qobj(
            self.to_sparse(), dims=[list(self.hilbert.shape), list(self.hilbert.shape)]
        )

    def __call__(self, v: np.ndarray) -> np.ndarray:
        return self.apply(v)

    def apply(self, v: np.ndarray) -> np.ndarray:
        op = self.to_linear_operator()
        return op @ v

    def __matmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or issparse(other)
        ):
            return self.apply(other)
        elif isinstance(other, AbstractOperator):
            return self._op__matmul__(other)
        else:
            return NotImplemented

    def _op__matmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented

    def __rmatmul__(self, other):
        if (
            isinstance(other, np.ndarray)
            or isinstance(other, jnp.ndarray)
            or issparse(other)
        ):
            # return self.apply(other)
            return NotImplemented
        elif isinstance(other, AbstractOperator):
            return self._op__rmatmul__(other)
        else:
            return NotImplemented

    def _op__rmatmul__(self, other):
        "Implementation on subclasses of __matmul__"
        return NotImplemented

    def to_linear_operator(self):
        return self.to_sparse()
