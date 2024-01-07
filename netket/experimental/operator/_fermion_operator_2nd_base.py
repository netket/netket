# Copyright 2022 The NetKet Authors - All rights reserved.
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

from typing import Union, Optional

import numpy as np
from numbers import Number

from netket.utils.types import DType
from netket.operator._discrete_operator import DiscreteOperator
from netket.operator._pauli_strings.base import _count_of_locations
from netket.hilbert.abstract_hilbert import AbstractHilbert
from netket.utils.numbers import is_scalar, dtype as _dtype
from netket.utils.optional_deps import import_optional_dependency

from netket.experimental.hilbert import SpinOrbitalFermions

from ._fermion_operator_2nd_utils import (
    _convert_terms_to_spin_blocks,
    _collect_constants,
    _canonicalize_input,
    _check_hermitian,
    _herm_conj,
    _make_tuple_tree,
    _remove_dict_zeros,
    _verify_input,
    _reduce_operators,
    OperatorDict,
    OperatorTermsList,
    OperatorWeightsList,
    _normal_ordering,
    _pair_ordering,
)


class FermionOperator2ndBase(DiscreteOperator):
    r"""
    A fermionic operator in :math:`2^{nd}` quantization.

    This is a base class that only provides symbolic-level manipulation capabilities,
    and must be inherited from a subclass defining indexing methods.
    """

    def __init__(
        self,
        hilbert: AbstractHilbert,
        terms: Union[list[str], list[list[list[int]]]] = None,
        weights: Optional[list[Union[float, complex]]] = None,
        constant: Union[float, complex] = 0.0,
        dtype: DType = None,
    ):
        r"""
        Constructs a fermion operator given the single terms (set of
        creation/annihilation operators) in second quantization formalism.

        This class can be initialized in the following form:
        `FermionOperator2nd(hilbert, terms, weights ...)`.
        The term contains pairs of `(idx, dagger)`, where `idx ∈ range(hilbert.size)`
        (it identifies an orbital) and `dagger` is a True/False flag determining if the
        operator is a creation or destruction operator.
        A term of the form :math:`\hat{a}_1^\dagger \hat{a}_2` would take the form
        `((1,1), (2,0))`, where (1,1) represents :math:`\hat{a}_1^\dagger` and (2,0)
        represents :math:`\hat{a}_2`.
        To split up per spin, use the creation and annihilation operators to build the
        operator.

        Args:
            hilbert: hilbert of the resulting FermionOperator2nd object
            terms: single term operators (see
                example below)
            weights: corresponding coefficients of the single term operators
                (defaults to a list of 1)
            constant: constant contribution, corresponding to the
                identity operator * constant (default = 0)

        Returns:
            A FermionOperator2nd object.

        Example:
            Constructs the fermionic hamiltonian in :math:`2^{nd}` quantization
            :math:`(0.5-0.5j)*(a_0^\dagger a_1) + (0.5+0.5j)*(a_2^\dagger a_1)`.

            >>> import netket.experimental as nkx
            >>> terms, weights = (((0,1),(1,0)),((2,1),(1,0))), (0.5-0.5j,0.5+0.5j)
            >>> hi = nkx.hilbert.SpinOrbitalFermions(3)
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> terms = ("0^ 1", "2^ 1")
            >>> op = nkx.operator.FermionOperator2nd(hi, terms, weights)
            >>> op
            FermionOperator2nd(hilbert=SpinOrbitalFermions(n_orbitals=3), n_operators=2, dtype=complex128)
            >>> op.hilbert
            SpinOrbitalFermions(n_orbitals=3)
            >>> op.hilbert.size
            3

        """
        super().__init__(hilbert)

        # bring terms, weights into consistent form, autopromote dtypes if necessary
        _operators, _constant, dtype = _canonicalize_input(
            terms, weights, constant, dtype
        )
        _verify_input(hilbert, _operators, raise_error=True)
        self._dtype = dtype

        # we keep the input, in order to be able to add terms later
        self._operators = _operators
        self._constant = _constant

        self._initialized = False
        self._is_hermitian = None  # set when requested
        self._max_conn_size = None

    def _reset_caches(self):
        """
        Cleans the internal caches built on the operator.
        """
        self._initialized = False
        self._is_hermitian = None
        self._max_conn_size = None

    def _setup(self, force: bool = False):
        """Analyze the operator strings and precompute arrays for get_conn inference"""
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def from_openfermion(
        cls,
        hilbert: AbstractHilbert,
        of_fermion_operator=None,  # : "openfermion.ops.FermionOperator" type
        *,
        n_orbitals: Optional[int] = None,
        convert_spin_blocks: bool = False,
    ) -> "FermionOperator2ndBase":
        r"""
        Converts an openfermion FermionOperator into a netket FermionOperator2nd.

        The hilbert first argument can be dropped, see __init__ for details and default
        value.
        Warning: convention of openfermion.hamiltonians is different from ours: instead
        of strong spin components as subsequent hilbert state outputs (i.e. the 1/2 spin
        components of spin-orbit i are stored in locations (2*i, 2*i+1)), we concatenate
        blocks of definite spin (i.e. locations (i, n_orbitals+i)).

        Args:
            hilbert: (optional) hilbert of the resulting FermionOperator2nd object
            of_fermion_operator (openfermion.ops.FermionOperator):
                `FermionOperator object <https://quantumai.google/reference/python/openfermion/ops/FermionOperator>`_ .
                More information about those objects can be found in
                `OpenFermion's documentation <https://quantumai.google/reference/python/openfermion>`_
            n_orbitals: (optional) total number of orbitals in the system, default
                None means inferring it from the FermionOperator2nd. Argument is
                ignored when hilbert is given.
            convert_spin_blocks: whether or not we need to convert the FermionOperator
                to our convention. Only works if hilbert is provided and if it has
                spin != 0

        """
        openfermion = import_optional_dependency(
            "openfermion", descr="from_openfermion"
        )
        FermionOperator = openfermion.ops.FermionOperator

        if hilbert is None:
            raise ValueError(
                "The first argument `from_openfermion` must either be an "
                "openfermion operator or an Hilbert space, followed by "
                "an openfermion operator"
            )

        if not isinstance(hilbert, AbstractHilbert):
            # if first argument is not Hilbert, then shift all arguments by one
            hilbert, of_fermion_operator = None, hilbert

        if not isinstance(of_fermion_operator, FermionOperator):  # pragma: no cover
            raise NotImplementedError()

        if convert_spin_blocks and hilbert is None:
            raise ValueError("if convert_spin_blocks, the hilbert must be specified")

        terms = list(of_fermion_operator.terms.keys())
        weights = list(of_fermion_operator.terms.values())
        terms, weights, constant = _collect_constants(terms, weights)

        if hilbert is not None:
            # no warning, just overwrite
            n_orbitals = hilbert.n_orbitals

            if convert_spin_blocks:
                if not hasattr(hilbert, "spin") or hilbert.spin is None:
                    raise ValueError(
                        f"cannot convert spin blocks for hilbert space {hilbert} without spin"
                    )
                n_spin = hilbert._n_spin_states
                terms = _convert_terms_to_spin_blocks(terms, n_orbitals, n_spin)
        if n_orbitals is None:
            # we always start counting from 0, so we only determine the maximum location
            n_orbitals = _count_of_locations(of_fermion_operator)
        if hilbert is None:
            hilbert = SpinOrbitalFermions(n_orbitals)  # no spin splitup assumed

        return cls(hilbert, terms, weights=weights, constant=constant)

    def __repr__(self):
        return (
            f"{type(self).__name__}(hilbert={self.hilbert}, "
            f"n_operators={len(self._operators)}, dtype={self.dtype})"
        )

    def reduce(self, order: bool = True, inplace: bool = True):
        """Prunes the operator by removing all terms with zero weights, grouping, and normal ordering (inplace)."""
        operators = self._operators
        terms, weights = list(operators.keys()), list(operators.values())

        if order:
            terms, weights = _normal_ordering(terms, weights)

        terms, weights, constant = _collect_constants(terms, weights)
        operators = dict(zip(terms, weights))

        self._operators = _reduce_operators(operators, self.dtype)
        self._constant = self._constant + constant

    @property
    def dtype(self) -> DType:
        """The dtype of the operator's matrix elements ⟨σ|Ô|σ'⟩."""
        return self._dtype

    @property
    def terms(self) -> OperatorTermsList:
        """Returns the list of all terms in the tuple notation."""
        return list(self._operators.keys())

    @property
    def weights(self) -> OperatorWeightsList:
        """Returns the list of the weights correspoding to the operator terms."""
        return list(self._operators.values())

    @property
    def constant(self) -> Number:
        """Returns the operator constant term."""
        return self._constant

    @property
    def operators(self) -> OperatorDict:
        """Returns a dictionary with (term, weight) key-value pairs, with terms in tuple notation. Does not include the constant."""
        return self._operators

    def copy(self, *, dtype: Optional[DType] = None):
        """
        Creates a deep copy of this operator, potentially changing the dtype of the
        operator and internal arrays.

        Args:
            dtype: Optional new dtype. Must be compatible with the current dtype.

        Returns:
            An identical operator that does not reference the internal arrays of
            the original one.
        """
        if dtype is None:
            dtype = self.dtype
        if not np.can_cast(self.dtype, dtype, casting="same_kind"):
            raise ValueError(f"Cannot cast {self.dtype} to {dtype}")

        op = type(self)(self.hilbert, constant=self._constant, dtype=dtype)

        if dtype == self.dtype:
            operators_new = self._operators.copy()
        else:
            operators_new = {
                k: np.array(v, dtype=dtype) for k, v in self._operators.items()
            }

        op._operators = operators_new
        return op

    def _remove_zeros(self):
        """Reduce the number of operators by removing unnecessary zeros"""
        op = type(self)(self.hilbert, constant=self._constant, dtype=self.dtype)
        op._operators = _remove_dict_zeros(self._operators)
        return op

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        if self._max_conn_size is None:
            self._setup()
        return self._max_conn_size

    @property
    def is_hermitian(self) -> bool:
        """Returns true if this operator is hermitian."""
        if self._is_hermitian is None:  # only compute when needed, is expensive
            terms = list(self._operators.keys())
            weights = list(self._operators.values())
            self._is_hermitian = _check_hermitian(terms, weights)
        return self._is_hermitian

    def operator_string(self) -> str:
        """Return a readable string describing all the operator terms"""
        op_string = []
        if not np.isclose(self._constant, 0.0):
            op_string.append(f"{self._constant} []")
        for term, weight in self._operators.items():
            s = []
            for idx, dag in term:
                dag_string = "^" if bool(dag) else ""
                s.append(f"{int(idx)}{dag_string}")
            s = " ".join(s)
            s = f"{weight} [{s}]"
            op_string.append(s)
        return " +\n".join(op_string)

    def _op__imatmul__(self, other):
        if not isinstance(other, FermionOperator2ndBase):  # pragma: no cover
            return NotImplemented
        if not self.hilbert == other.hilbert:
            raise ValueError(
                "Can only multiply identical hilbert spaces (got A@B, "
                f"A={self.hilbert}, B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )

        new_operators = {}
        for t, w in self._operators.items():
            for to, wo in other._operators.items():
                # if the last operator of t and the first of to are
                # equal, we have a ...ĉᵢĉᵢ... which is null.
                if t[-1] != to[0]:
                    new_t = t + to
                    new_operators[new_t] = new_operators.get(new_t, 0) + w * wo

        if not np.isclose(other._constant, 0.0):
            for t, w in self._operators.items():
                new_operators[t] = w * other._constant
        if not np.isclose(self._constant, 0.0):
            for t, w in other._operators.items():
                new_operators[t] = w * self._constant

        self._operators = new_operators
        self._constant = self._constant * other._constant
        self._reset_caches()
        return self

    def _op__matmul__(self, other):
        if not isinstance(other, FermionOperator2ndBase):  # pragma: no cover
            return NotImplemented
        dtype = np.promote_types(self.dtype, other.dtype)
        op = self.copy(dtype=dtype)
        return op._op__imatmul__(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):
        dtype = np.promote_types(self.dtype, _dtype(other))
        op = self.copy(dtype=dtype)
        return op.__iadd__(other)

    def __iadd__(self, other):
        if is_scalar(other):
            if not np.isclose(other, 0.0):
                self._constant += other
            return self
        if not isinstance(other, FermionOperator2ndBase):  # pragma: no cover
            raise NotImplementedError(
                f"In-place addition not implemented for {type(self)} "
                f"and {type(other)}"
            )
        if not self.hilbert == other.hilbert:
            raise ValueError(
                f"Can only add identical hilbert spaces (got A+B, A={self.hilbert}, "
                f"B={other.hilbert})"
            )
        if not np.can_cast(_dtype(other), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot add inplace operator with dtype {type(other)} "
                f"to operator with dtype {self.dtype}"
            )
        self_ops = self._operators
        for t, w in other._operators.items():
            sw = self_ops.get(t, None)
            if sw is None and not np.isclose(w, 0):
                self_ops[t] = w
            elif sw is not None:
                w = sw + w
                if np.isclose(w, 0):
                    del self_ops[t]
                else:
                    self_ops[t] = w

        self._constant += other._constant
        self._reset_caches()
        return self

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self.__mul__(np.array(-1, dtype=self.dtype))

    def __rmul__(self, scalar):
        return self * scalar

    def __imul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            self._op__imatmul__(scalar)
        if not np.can_cast(_dtype(scalar), self.dtype, casting="same_kind"):
            raise ValueError(
                f"Cannot multiply inplace scalar with dtype {type(scalar)} "
                f"to operator with dtype {self.dtype}"
            )
        scalar = np.array(scalar, dtype=self.dtype).item()

        if np.isclose(scalar, 0):
            new_operators = {}
        else:
            new_operators = {o: scalar * v for o, v in self._operators.items()}

        self._operators = new_operators
        self._constant *= scalar
        self._reset_caches()
        return self

    def __mul__(self, scalar):
        if not is_scalar(scalar):
            # we will overload this as matrix multiplication
            return self._op__matmul__(scalar)
        dtype = np.promote_types(self.dtype, _dtype(scalar))
        op = self.copy(dtype=dtype)
        return op.__imul__(scalar)

    def conjugate(self, *, concrete=False):
        r"""Returns the complex conjugate of this operator."""

        terms = list(self._operators.keys())
        weights = list(self._operators.values())
        terms, weights = _herm_conj(terms, weights)  # changes also the terms
        terms = _make_tuple_tree(terms)

        cls = type(self)
        new = cls(
            self.hilbert,
            constant=np.conjugate(self._constant),
            dtype=self.dtype,
        )
        new._operators = dict(zip(terms, weights))
        return new

    def to_normal_order(self):
        """Reoder the operators to normal order.
        Normal ordering corresponds to placing creating operators on the left and annihilation on the right.
        Then, it places the highest index on the left and lowest index on the right
        In this ordering, we make sure to account for the anti-commutation of operators.
        `Normal ordering documentation <https://en.wikipedia.org/wiki/Normal_order#Fermions>`_
        """
        terms, weights = _normal_ordering(self.terms, self.weights)
        new = type(self)(
            self.hilbert,
            constant=self._constant,
            dtype=self.dtype,
        )
        new._operators = dict(zip(terms, weights))
        new.reduce()
        return new

    def to_pair_order(self):
        """Reoder the operators to pair order.
        Pair ordering corresponds to placing first the highest indices on the right,
        and then making sure the creation operators are on the left and annihilation on the right.
        In this ordering, we make sure to account for the anti-commutation of operators.
        """
        terms, weights = _pair_ordering(self.terms, self.weights)
        new = type(self)(
            self.hilbert,
            constant=self._constant,
            dtype=self.dtype,
        )
        new._operators = dict(zip(terms, weights))
        new.reduce()
        return new
