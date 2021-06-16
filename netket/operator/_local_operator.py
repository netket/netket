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

import numbers
from typing import Union, List, Optional
from netket.utils.types import DType, Array
from textwrap import dedent

import numpy as np
from numba import jit

from netket.hilbert import AbstractHilbert, Fock

from ._abstract_operator import AbstractOperator
from ._lazy import Transpose


@jit(nopython=True)
def _number_to_state(number, hilbert_size_per_site, local_states_per_site, out):

    out[:] = local_states_per_site[:, 0]
    size = out.shape[0]

    ip = number
    k = size - 1
    while ip > 0:
        local_size = hilbert_size_per_site[k]
        out[k] = local_states_per_site[k, ip % local_size]
        ip = ip // local_size
        k -= 1

    return out


def is_hermitian(a: np.ndarray, rtol=1e-05, atol=1e-08) -> bool:
    return np.allclose(a, a.T.conj(), rtol=rtol, atol=atol)


def _dtype(obj: Union[numbers.Number, Array, "LocalOperator"]) -> DType:
    if isinstance(obj, numbers.Number):
        return type(obj)
    elif isinstance(obj, AbstractOperator):
        return obj.dtype
    elif isinstance(obj, np.ndarray):
        return obj.dtype
    else:
        raise TypeError(f"cannot deduce dtype of object type {type(obj)}: {obj}")


def has_nonzero_diagonal(op: "LocalOperator") -> bool:
    """
    Returns True if at least one element in the diagonal of the operator
    is nonzero.
    """
    return (
        np.any(np.abs(op._diag_mels) >= op.mel_cutoff)
        or np.abs(op._constant) >= op.mel_cutoff
    )


def _is_sorted(a):
    for i in range(len(a) - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def resize(
    arr: Array,
    shape: List[int],
    dtype: Optional[DType] = None,
    init: Optional[numbers.Number] = None,
) -> Array:
    """
    resizes the input array to the new shape that must be larger than the old.

    The resulting array is initialized with the old array in the corresponding indices, and with init
    in the rest.

    Args:
        arr: The array to be resized
        shape: The new shape
        dtype: optional dtype of the new array. If unspecified the old array dtype is used
        init: optional initialization value for the new entries

    Returns:
        a numpy array with the chosen shape and dtype.
    """
    if dtype is None:
        dtype = arr.dtype

    if isinstance(shape, int):
        shape = (shape,)

    if arr.shape == shape:
        return arr

    arr_shape = arr.shape
    if len(shape) != arr.ndim:
        raise ValueError("the number of dimensions should not change.")

    for (i, ip) in zip(arr_shape, shape):
        if ip < i:
            raise ValueError(
                f"The new dimensions ({shape}) should all be larger than the old ({arr_shape})."
            )

    new_arr = np.empty(shape=shape, dtype=arr.dtype)
    if init is not None:
        new_arr[...] = init

    if arr.ndim == 0:
        raise ValueError("Cannot resize a 0-dimensional scalar")
    elif arr.ndim == 1:
        new_arr[: arr_shape[0]] = arr
    elif arr.ndim == 2:
        new_arr[: arr_shape[0], : arr_shape[1]] = arr
    elif arr.ndim == 3:
        new_arr[: arr_shape[0], : arr_shape[1], : arr_shape[2]] = arr
    elif arr.ndim == 4:
        new_arr[: arr_shape[0], : arr_shape[1], : arr_shape[2], : arr_shape[3]] = arr
    else:
        raise ValueError(f"unsupported number of dimensions: {arr.ndim}")

    return new_arr


def _reorder_kronecker_product(hi, mat, acting_on):
    """
    Reorders the matrix resulting from a kronecker product of several
    operators in such a way to sort acting_on.

    A conceptual example is the following:
    if `mat = Â ⊗ B̂ ⊗ Ĉ` and `acting_on = [[2],[1],[3]`
    you will get `result = B̂ ⊗ Â ⊗ Ĉ, [[1], [2], [3]].

    However, essentially, A,B,C represent some operators acting on
    thei sub-space acting_on[1], [2] and [3] of the hilbert space.

    This function also handles any possible set of values in acting_on.

    The inner logic uses the Fock.all_states(), number_to_state and
    state_to_number to perform the re-ordering.
    """
    acting_on_sorted = np.sort(acting_on)
    if np.all(acting_on_sorted == acting_on):
        return mat, acting_on

    # could write custom binary <-> int logic instead of using Fock...
    # Since i need to work with bit-strings (where instead of bits i
    # have integers, in order to support arbitrary size spaces) this
    # is exactly what hilbert.to_number() and viceversa do.

    # target ordering binary representation
    hi_subspace = Fock(hi.shape[acting_on_sorted[0]] - 1)
    for site in acting_on_sorted[1:]:
        hi_subspace = Fock(hi.shape[site] - 1) * hi_subspace

    # find how to map target ordering back to unordered
    acting_on_unsorted_ids = np.zeros(len(acting_on), dtype=np.intp)
    for (i, site) in enumerate(acting_on):
        acting_on_unsorted_ids[i] = np.argmax(site == acting_on_sorted)

    # now it is valid that
    # acting_on_sorted == acting_on[acting_on_unsorted_ids]

    # generate n-bit strings in the target ordering
    v = hi_subspace.all_states()

    # convert them to origin (unordered) ordering
    v_unsorted = v[:, acting_on_unsorted_ids]
    # convert the unordered bit-strings to numbers in the target space.
    n_unsorted = hi_subspace.states_to_numbers(v_unsorted)

    # reorder the matrix
    mat_sorted = mat[n_unsorted, :][:, n_unsorted]

    return mat_sorted, acting_on_sorted


class LocalOperator(AbstractOperator):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[List[Array], Array] = [],
        acting_on: Union[List[int], List[List[int]]] = [],
        constant: numbers.Number = 0,
        dtype: Optional[DType] = None,
    ):
        r"""
        Constructs a new ``LocalOperator`` given a hilbert space and (if
        specified) a constant level shift.

        Args:
           hilbert (netket.AbstractHilbert): Hilbert space the operator acts on.
           operators (list(numpy.array) or numpy.array): A list of operators, in matrix form.
           acting_on (list(numpy.array) or numpy.array): A list of sites, which the corresponding operators act on.
           constant (float): Level shift for operator. Default is 0.0.

        Examples:
           Constructs a ``LocalOperator`` without any operators.

           >>> from netket.hilbert import CustomHilbert
           >>> from netket.operator import LocalOperator
           >>> hi = CustomHilbert(local_states=[-1, 1])**20
           >>> empty_hat = LocalOperator(hi)
           >>> print(len(empty_hat.acting_on))
           0
        """
        super().__init__(hilbert)
        self._constant = constant

        if not all(
            [_is_sorted(hilbert.states_at_index(i)) for i in range(hilbert.size)]
        ):
            raise ValueError(
                dedent(
                    """LocalOperator needs an hilbert space with sorted state values at
                every site.
                """
                )
            )

        # check if passing a single operator or a list of operators
        if isinstance(acting_on, numbers.Number):
            acting_on = [acting_on]

        is_nested = any(hasattr(i, "__len__") for i in acting_on)

        if not is_nested:
            operators = [operators]
            acting_on = [acting_on]

        operators = [np.asarray(operator) for operator in operators]

        # If we asked for a specific dtype, enforce it.
        if dtype is None:
            dtype = np.promote_types(operators[0].dtype, np.float32)
            for op in operators[1:]:
                np.promote_types(dtype, op.dtype)

        self._dtype = dtype
        self._init_zero()

        self.mel_cutoff = 1.0e-6

        self._nonzero_diagonal = np.abs(self._constant) >= self.mel_cutoff
        """True if at least one element in the diagonal of the operator is
        nonzero"""

        for op, act in zip(operators, acting_on):
            if len(act) > 0:
                self._add_operator(op, act)

    @property
    def operators(self) -> List[np.ndarray]:
        """List of the matrices of the operators encoded in this Local Operator.
        Returns a copy.
        """
        return self._operators_list()

    @property
    def acting_on(self) -> List[List[int]]:
        """List containing the list of the sites on which every operator acts.

        Every operator `self.operators[i]` acts on the sites `self.acting_on[i]`
        """
        actions = [action[action >= 0].tolist() for action in self._acting_on]
        return actions

    @property
    def dtype(self) -> DType:
        return self._dtype

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        return self._is_hermitian

    @property
    def mel_cutoff(self) -> float:
        r"""float: The cutoff for matrix elements.
        Only matrix elements such that abs(O(i,i))>mel_cutoff
        are considered"""
        return self._mel_cutoff

    @mel_cutoff.setter
    def mel_cutoff(self, mel_cutoff):
        self._mel_cutoff = mel_cutoff
        assert self.mel_cutoff >= 0

    @property
    def constant(self) -> numbers.Number:
        return self._constant

    @property
    def n_operators(self) -> int:
        return self._n_operators

    def __add__(self, other: Union["LocalOperator", numbers.Number]):
        op = self.copy(dtype=np.promote_types(self.dtype, _dtype(other)))
        op = op.__iadd__(other)
        return op

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, LocalOperator):
            if self.hilbert != other.hilbert:
                return NotImplemented

            if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {other.dtype} to operator with dtype {self.dtype}"
                )

            assert other.mel_cutoff == self.mel_cutoff

            for i in range(other._n_operators):
                acting_on = other._acting_on[i, : other._acting_size[i]]
                operator = other._operators[i]
                self._add_operator(operator, acting_on)

            self._constant += other.constant
            self._nonzero_diagonal = has_nonzero_diagonal(self)

            return self
        if isinstance(other, numbers.Number):

            if not np.can_cast(type(other), self.dtype, casting="same_kind"):
                raise ValueError(
                    f"Cannot add inplace operator with dtype {type(other)} to operator with dtype {self.dtype}"
                )

            self._constant += other
            self._nonzero_diagonal = has_nonzero_diagonal(self)
            return self

        return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __isub__(self, other):
        return self.__iadd__(-other)

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if isinstance(other, AbstractOperator):
            op = self.copy(dtype=np.promote_types(self.dtype, _dtype(other)))
            return op.__imatmul__(other)
        elif not isinstance(other, numbers.Number):
            return NotImplemented

        op = self.copy(dtype=np.promote_types(self.dtype, _dtype(other)))

        op._diag_mels *= other
        op._mels *= other
        op._constant *= other

        for _op in op._operators:
            _op *= other

        op._nonzero_diagonal = has_nonzero_diagonal(op)

        return op

    def __imul__(self, other):
        if isinstance(other, AbstractOperator):
            return self.__imatmul__(other)
        elif not isinstance(other, numbers.Number):
            return NotImplemented

        if not np.can_cast(type(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} to operator with dtype {self.dtype}"
            )

        self._diag_mels *= other
        self._mels *= other
        self._constant *= other

        for _op in self._operators:
            _op *= other

        self._nonzero_diagonal = has_nonzero_diagonal(self)

        return self

    def __imatmul__(self, other):
        if not isinstance(other, LocalOperator):
            return NotImplemented

        if not np.can_cast(other.dtype, self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} to operator with dtype {self.dtype}"
            )

        return self._concrete_imatmul_(other)

    def _op__matmul__(self, other):
        return self._concrete_matmul_(other)

    def _concrete_matmul_(self, other: "LocalOperator") -> "LocalOperator":
        if not isinstance(other, LocalOperator):
            return NotImplemented
        op = self.copy(dtype=np.promote_types(self.dtype, _dtype(other)))
        op @= other
        return op

    def _concrete_imatmul_(self, other: "LocalOperator") -> "LocalOperator":
        if not isinstance(other, LocalOperator):
            return NotImplemented

        tot_operators = []
        tot_act = []
        for i in range(other._n_operators):
            act_i = other._acting_on[i, : other._acting_size[i]].tolist()
            ops, act = self._multiply_operator(other._operators[i], act_i)
            tot_operators += ops
            tot_act += act

        prod = LocalOperator(self.hilbert, tot_operators, tot_act, dtype=self.dtype)
        self_constant = self._constant
        if np.abs(other._constant) > self.mel_cutoff:
            self *= other._constant
            self += prod
            self._constant = 0.0
        else:
            self = prod

        if np.abs(self_constant) > self.mel_cutoff:
            self += other * self_constant

        self._nonzero_diagonal = has_nonzero_diagonal(self)

        return self

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            raise TypeError("Only divison by a scalar number is supported.")

        if other == 0:
            raise ValueError("Dividing by 0")
        return self.__mul__(1.0 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def _init_zero(self):
        self._operators = []
        self._n_operators = 0

        self._max_op_size = 0
        self._max_acting_size = 0
        self._max_local_hilbert_size = 0
        self._size = 0

        self._acting_on = np.zeros((0, 0), dtype=np.intp)
        self._acting_size = np.zeros(0, dtype=np.intp)
        self._diag_mels = np.zeros((0, 0), dtype=self.dtype)
        self._mels = np.empty((0, 0, 0), dtype=self.dtype)
        self._x_prime = np.empty((0, 0, 0, 0))
        self._n_conns = np.empty((0, 0), dtype=np.intp)

        self._local_states = np.zeros((0, 0, 0), dtype=np.float64)

        self._basis = np.zeros((0, 0), dtype=np.int64)
        self._is_hermitian = True

    def _acting_on_list(self):
        acting_on = []
        for i in range(self.n_operators):
            acting_on.append(np.copy(self._acting_on[i, : self._acting_size[i]]))

        return acting_on

    def _operators_list(self):
        "A deep copy of the operators"
        operators = [np.copy(op) for op in self._operators]
        return operators

    def _add_operator(self, operator: Array, acting_on: List[int]):
        if not np.can_cast(operator, self.dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast type {operator.dtype} to {self.dtype}")

        acting_on = np.asarray(acting_on, dtype=np.intp)
        operator = np.asarray(operator, dtype=self.dtype)

        if np.unique(acting_on).size != acting_on.size:
            raise ValueError("acting_on contains repeated entries.")

        if any(acting_on >= self.hilbert.size):
            raise ValueError("acting_on points to a site not in the hilbert space.")

        if operator.ndim != 2:
            raise ValueError("The operator should be a matrix")

        if np.all(np.abs(operator) < self.mel_cutoff):
            return

        # re-sort the operator
        operator, acting_on = _reorder_kronecker_product(
            self.hilbert, operator, acting_on
        )

        # find overlapping support
        support_i = None
        for (i, support) in enumerate(self._acting_on_list()):
            if np.all(acting_on == support):
                support_i = i
                break

        # If overlapping support, add the local operators themselves
        if support_i is not None:
            dim = min(operator.shape[0], self._operators[support_i].shape[0])
            _opv = self._operators[support_i][:dim, :dim]
            _opv += operator[:dim, :dim]

            n_local_states_per_site = np.asarray(
                [self.hilbert.size_at_index(i) for i in acting_on]
            )

            self._append_matrix(
                self._operators[support_i],
                self._diag_mels[support_i],
                self._mels[support_i],
                self._x_prime[support_i],
                self._n_conns[support_i],
                self._acting_size[support_i],
                self._local_states[support_i],
                self.mel_cutoff,
                n_local_states_per_site,
            )

            isherm = True
            for op in self._operators:
                isherm = isherm and is_hermitian(op)

            self._is_hermitian = isherm
            self._nonzero_diagonal = has_nonzero_diagonal(self)
        else:
            self.__add_new_operator__(operator, acting_on)

    def __add_new_operator__(self, operator: np.ndarray, acting_on: np.ndarray):
        # Else, we must add a completely new operator
        self._n_operators += 1
        self._operators.append(operator)

        # Add a new row and eventually resize the acting_on
        self._acting_size = np.resize(self._acting_size, (self.n_operators,))
        self._acting_size[-1] = acting_on.size
        acting_size = acting_on.size

        self._max_op_size = max((operator.shape[0], self._max_op_size))

        n_local_states_per_site = np.asarray(
            [self.hilbert.size_at_index(i) for i in acting_on]
        )

        if operator.shape[0] != np.prod(n_local_states_per_site):
            raise RuntimeError(
                r"""the given operator matrix has shape={} and acts on
                    the sites={}, which have a local hilbert space size of
                    sizes={} giving an expected shape
                    for the operator expected_shape={}.""".format(
                    operator.shape,
                    acting_on,
                    n_local_states_per_site,
                    np.prod(n_local_states_per_site),
                )
            )

        self._max_acting_size = max(self._max_acting_size, acting_on.size)
        self._max_local_hilbert_size = max(
            self._max_local_hilbert_size, np.max(n_local_states_per_site)
        )

        self._acting_on = resize(
            self._acting_on, shape=(self.n_operators, self._max_acting_size), init=-1
        )
        self._acting_on[-1, :acting_size] = acting_on
        if (
            self._acting_on[-1, :acting_size].max() > self.hilbert.size
            or self._acting_on[-1, :acting_size].min() < 0
        ):
            raise ValueError("Operator acts on an invalid set of sites")

        self._local_states = resize(
            self._local_states,
            shape=(
                self.n_operators,
                self._max_acting_size,
                self._max_local_hilbert_size,
            ),
            init=np.nan,
        )
        ## add an operator to local_states
        for site in range(acting_size):
            self._local_states[-1, site, : n_local_states_per_site[site]] = np.asarray(
                self.hilbert.states_at_index(acting_on[site])
            )
        ## add an operator to basis
        self._basis = resize(
            self._basis, shape=(self.n_operators, self._max_acting_size), init=1e10
        )
        ba = 1
        for s in range(acting_on.size):
            self._basis[-1, s] = ba
            ba *= n_local_states_per_site[acting_on.size - s - 1]
        ##

        self._diag_mels = resize(
            self._diag_mels, shape=(self.n_operators, self._max_op_size), init=np.nan
        )
        self._mels = resize(
            self._mels,
            shape=(self.n_operators, self._max_op_size, self._max_op_size - 1),
            init=np.nan,
        )
        self._x_prime = resize(
            self._x_prime,
            shape=(
                self.n_operators,
                self._max_op_size,
                self._max_op_size - 1,
                self._max_acting_size,
            ),
            init=-1,
        )
        self._n_conns = resize(
            self._n_conns, shape=(self.n_operators, self._max_op_size), init=-1
        )
        if acting_on.max() + 1 >= self._size:
            self._size = acting_on.max() + 1

        self._append_matrix(
            operator,
            self._diag_mels[-1],
            self._mels[-1],
            self._x_prime[-1],
            self._n_conns[-1],
            self._acting_size[-1],
            self._local_states[-1],
            self.mel_cutoff,
            n_local_states_per_site,
        )

        isherm = True
        for op in self._operators:
            isherm = isherm and is_hermitian(op)

        self._is_hermitian = isherm

        self._nonzero_diagonal = has_nonzero_diagonal(self)

    @staticmethod
    @jit(nopython=True)
    def _append_matrix(
        operator,
        diag_mels,
        mels,
        x_prime,
        n_conns,
        acting_size,
        local_states_per_site,
        epsilon,
        hilb_size_per_site,
    ):
        op_size = operator.shape[0]
        assert op_size == operator.shape[1]
        for i in range(op_size):
            diag_mels[i] = operator[i, i]
            n_conns[i] = 0
            for j in range(op_size):
                if i != j and np.abs(operator[i, j]) > epsilon:
                    k_conn = n_conns[i]
                    mels[i, k_conn] = operator[i, j]
                    _number_to_state(
                        j,
                        hilb_size_per_site,
                        local_states_per_site[:acting_size, :],
                        x_prime[i, k_conn, :acting_size],
                    )
                    n_conns[i] += 1

    def _multiply_operator(self, op, act):
        operators = []
        acting_on = []
        act = np.asarray(act)

        for i in range(self.n_operators):
            act_i = self._acting_on[i, : self._acting_size[i]]

            inters = np.intersect1d(act_i, act, return_indices=False)

            if act.size == act_i.size and np.array_equal(act, act_i):
                # non-interesecting with same support
                operators.append(np.copy(np.matmul(self._operators[i], op)))
                acting_on.append(act_i.tolist())
            elif inters.size == 0:
                # disjoint supports
                operators.append(np.copy(np.kron(self._operators[i], op)))
                acting_on.append(act_i.tolist() + act.tolist())
            else:
                _act = list(act)
                _act_i = list(act_i)
                _op = op.copy()
                _op_i = self._operators[i].copy()

                # expand _act to match _act_i
                actmin = min(act)
                for site in act_i:
                    if site not in act:
                        I = np.eye(self.hilbert.shape[site], dtype=self.dtype)
                        if site < actmin:
                            _act = [site] + _act
                            _op = np.kron(I, _op)
                        else:  #  site > actmax
                            _act = _act + [site]
                            _op = np.kron(_op, I)

                act_i_min = min(act_i)
                for site in act:
                    if site not in act_i:
                        I = np.eye(self.hilbert.shape[site], dtype=self.dtype)
                        if site < act_i_min:
                            _act_i = [site] + _act_i
                            _op_i = np.kron(I, _op_i)
                        else:  #  site > actmax
                            _act_i = _act_i + [site]
                            _op_i = np.kron(_op_i, I)

                # reorder
                _op, _act = _reorder_kronecker_product(self.hilbert, _op, _act)
                _op_i, _act_i = _reorder_kronecker_product(self.hilbert, _op_i, _act_i)

                if len(_act) == len(_act_i) and np.array_equal(_act, _act_i):
                    # non-interesecting with same support
                    operators.append(np.matmul(_op_i, _op))
                    acting_on.append(_act_i)
                else:
                    raise ValueError("Something failed")

        return operators, acting_on

    def copy(self, *, dtype: Optional = None):
        """Returns a copy of the operator, while optionally changing the dtype
        of the operator.

        Args:
            dtype: optional dtype
        """

        if dtype is None:
            dtype = self.dtype

        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")

        return LocalOperator(
            hilbert=self.hilbert,
            operators=[np.copy(op) for op in self._operators],
            acting_on=self._acting_on_list(),
            constant=self._constant,
            dtype=dtype,
        )

    def transpose(self, *, concrete=False):
        r"""LocalOperator: Returns the tranpose of this operator."""
        if concrete:

            new_ops = [np.copy(ops.transpose()) for ops in self._operators]

            return LocalOperator(
                hilbert=self.hilbert,
                operators=new_ops,
                acting_on=self._acting_on_list(),
                constant=self._constant,
            )
        else:
            return Transpose(self)

    def conjugate(self, *, concrete=False):
        r"""LocalOperator: Returns the complex conjugate of this operator."""
        new_ops = [np.copy(ops).conjugate() for ops in self._operators]

        return LocalOperator(
            hilbert=self.hilbert,
            operators=new_ops,
            acting_on=self._acting_on_list(),
            constant=np.conjugate(self._constant),
        )

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        max_size = self.n_operators if self._nonzero_diagonal else 0
        for op in self._operators:
            nnz_mat = np.abs(op) > self.mel_cutoff
            nnz_mat[np.diag_indices(nnz_mat.shape[0])] = False
            nnz_rows = np.sum(nnz_mat, axis=1)
            max_size += np.max(nnz_rows)

        return max_size

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            pad (bool): Whether to use zero-valued matrix elements in order to return all equal sections.

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._get_conn_flattened_kernel(
            np.asarray(x),
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            self._nonzero_diagonal,
            pad,
        )

    def _get_conn_flattened_closure(self):
        _local_states = self._local_states
        _basis = self._basis
        _constant = self._constant
        _diag_mels = self._diag_mels
        _n_conns = self._n_conns
        _mels = self._mels
        _x_prime = self._x_prime
        _acting_on = self._acting_on
        _acting_size = self._acting_size
        # workaround my painfully discovered Numba#6979 (cannot use numpy bools in closures)
        _nonzero_diagonal = bool(self._nonzero_diagonal)

        fun = self._get_conn_flattened_kernel

        def gccf_fun(x, sections):
            return fun(
                x,
                sections,
                _local_states,
                _basis,
                _constant,
                _diag_mels,
                _n_conns,
                _mels,
                _x_prime,
                _acting_on,
                _acting_size,
                _nonzero_diagonal,
            )

        return jit(nopython=True)(gccf_fun)

    @staticmethod
    @jit(nopython=True)
    def _get_conn_flattened_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        nonzero_diagonal,
        pad=False,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0
        max_conn = 0

        for b in range(batch_size):
            # diagonal element
            conn_b = 1 if nonzero_diagonal else 0

            # counting the off-diagonal elements
            for i in range(n_operators):
                acting_size_i = acting_size[i]

                xs_n[b, i] = 0
                x_b = x[b]
                x_i = x_b[acting_on[i, :acting_size_i]]
                for k in range(acting_size_i):
                    xs_n[b, i] += (
                        np.searchsorted(
                            local_states[i, acting_size_i - k - 1],
                            x_i[acting_size_i - k - 1],
                        )
                        * basis[i, k]
                    )

                conn_b += n_conns[i, xs_n[b, i]]

            tot_conn += conn_b
            sections[b] = tot_conn

            if pad:
                max_conn = max(conn_b, max_conn)

        if pad:
            tot_conn = batch_size * max_conn

        x_prime = np.empty((tot_conn, n_sites), dtype=x.dtype)
        mels = np.empty(tot_conn, dtype=dtype)

        c = 0
        for b in range(batch_size):
            c_diag = c
            x_batch = x[b]

            if nonzero_diagonal:
                mels[c_diag] = constant
                x_prime[c_diag] = np.copy(x_batch)
                c += 1

            for i in range(n_operators):

                # Diagonal part
                #  If nonzero_diagonal, this goes to c_diag = 0 ....
                # if zero_diagonal this just sets the last element to 0
                # so it's not worth it skipping it
                mels[c_diag] += diag_mels[i, xs_n[b, i]]
                n_conn_i = n_conns[i, xs_n[b, i]]

                if n_conn_i > 0:
                    sites = acting_on[i]
                    acting_size_i = acting_size[i]

                    for cc in range(n_conn_i):
                        mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                        x_prime[c + cc] = np.copy(x_batch)

                        for k in range(acting_size_i):
                            x_prime[c + cc, sites[k]] = all_x_prime[
                                i, xs_n[b, i], cc, k
                            ]
                    c += n_conn_i

            if pad:
                delta_conn = max_conn - (c - c_diag)
                mels[c : c + delta_conn].fill(0)
                x_prime[c : c + delta_conn, :] = np.copy(x_batch)
                c += delta_conn
                sections[b] = c

        return x_prime, mels

    def get_conn_filtered(self, x, sections, filters):
        r"""Finds the connected elements of the Operator using only a subset of operators. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                        the batch of quantum numbers x.
            sections (array): An array of size (batch_size) useful to unflatten
                        the output of this function.
                        See numpy.split for the meaning of sections.
            filters (array): Only operators op(filters[i]) are used to find the connected elements of
                        x[i].

        Returns:
            matrix: The connected states x', flattened together in a single matrix.
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._get_conn_filtered_kernel(
            x,
            sections,
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
            filters,
        )

    @staticmethod
    @jit(nopython=True)
    def _get_conn_filtered_kernel(
        x,
        sections,
        local_states,
        basis,
        constant,
        diag_mels,
        n_conns,
        all_mels,
        all_x_prime,
        acting_on,
        acting_size,
        filters,
    ):

        batch_size = x.shape[0]
        n_sites = x.shape[1]
        dtype = all_mels.dtype

        assert filters.shape[0] == batch_size and sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0

        for b in range(batch_size):
            # diagonal element
            tot_conn += 1

            # counting the off-diagonal elements
            i = filters[b]

            assert i < n_operators and i >= 0
            acting_size_i = acting_size[i]

            xs_n[b, i] = 0
            x_b = x[b]
            x_i = x_b[acting_on[i, :acting_size_i]]
            for k in range(acting_size_i):
                xs_n[b, i] += (
                    np.searchsorted(
                        local_states[i, acting_size_i - k - 1],
                        x_i[acting_size_i - k - 1],
                    )
                    * basis[i, k]
                )

            tot_conn += n_conns[i, xs_n[b, i]]
            sections[b] = tot_conn

        x_prime = np.empty((tot_conn, n_sites))
        mels = np.empty(tot_conn, dtype=dtype)

        c = 0
        for b in range(batch_size):
            c_diag = c
            mels[c_diag] = constant
            x_batch = x[b]
            x_prime[c_diag] = np.copy(x_batch)
            c += 1

            i = filters[b]
            # Diagonal part
            mels[c_diag] += diag_mels[i, xs_n[b, i]]
            n_conn_i = n_conns[i, xs_n[b, i]]

            if n_conn_i > 0:
                sites = acting_on[i]
                acting_size_i = acting_size[i]

                for cc in range(n_conn_i):
                    mels[c + cc] = all_mels[i, xs_n[b, i], cc]
                    x_prime[c + cc] = np.copy(x_batch)

                    for k in range(acting_size_i):
                        x_prime[c + cc, sites[k]] = all_x_prime[i, xs_n[b, i], cc, k]
                c += n_conn_i

        return x_prime, mels

    def __repr__(self):
        ao = self.acting_on

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str}, constant={self.constant}, dtype={self.dtype})"
