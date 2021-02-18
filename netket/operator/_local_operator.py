import numbers
from typing import Union, Tuple

import numpy as np
from numba import jit

from ._abstract_operator import AbstractOperator
from ._lazy import Transpose, Adjoint, Squared


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


def is_hermitian(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T.conj(), rtol=rtol, atol=atol)


class LocalOperator(AbstractOperator):
    """A custom local operator. This is a sum of an arbitrary number of operators
    acting locally on a limited set of k quantum numbers (i.e. k-local,
    in the quantum information sense).
    """

    def __init__(self, hilbert, operators=[], acting_on=[], constant=0):
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
           >>> hi = CustomHilbert(local_states=[1, -1])**20
           >>> empty_hat = LocalOperator(hi)
           >>> print(len(empty_hat.acting_on))
           0
        """
        super().__init__(hilbert)
        self._constant = constant

        self._init_zero()

        self.mel_cutoff = 1.0e-6

        # check if passing a single operator or a list of operators
        if isinstance(acting_on, numbers.Number):
            acting_on = [acting_on]

        is_nested = any(hasattr(i, "__len__") for i in acting_on)

        if not is_nested:
            operators = [operators]
            acting_on = [acting_on]

        for op, act in zip(operators, acting_on):
            if len(act) > 0:
                self._add_operator(op, act)

    @property
    def operators(self):
        return self._operators_list()

    @property
    def acting_on(self):
        actions = [action[action >= 0].tolist() for action in self._acting_on]
        return actions

    @property
    def size(self):
        return self._size

    @property
    def is_hermitian(self):
        return self._is_hermitian

    @property
    def mel_cutoff(self):
        r"""float: The cutoff for matrix elements.
        Only matrix elements such that abs(O(i,i))>mel_cutoff
        are considered"""
        return self._mel_cutoff

    @mel_cutoff.setter
    def mel_cutoff(self, mel_cutoff):
        self._mel_cutoff = mel_cutoff
        assert self.mel_cutoff >= 0

    @property
    def constant(self):
        return self._constant

    @property
    def n_operators(self):
        return self._n_conns.shape[0]

    def __add__(self, other: Union["LocalOperator", numbers.Number]):
        op = self.copy()
        op = op.__iadd__(other)
        return op

    def __radd__(self, other):
        return self.__add__(other)

    def __iadd__(self, other):
        if isinstance(other, LocalOperator):
            if self.hilbert != other.hilbert:
                return NotImplemented

            assert other.mel_cutoff == self.mel_cutoff

            for i in range(other._n_operators):
                acting_on = other._acting_on[i, : other._acting_size[i]]
                operator = other._operators[i]
                self._add_operator(operator, acting_on)

            self._constant += other._constant

            return self
        if isinstance(other, numbers.Number):
            self._constant += other
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
        op = self.copy()
        if isinstance(other, AbstractOperator):
            return op.__imatmul__(other)
        elif not isinstance(other, numbers.Number):
            return NotImplemented

        op._constant *= other
        op._diag_mels *= other
        op._mels *= other

        for _op in op._operators:
            _op *= other

        return op

    def __imatmul__(self, other):
        if not isinstance(other, LocalOperator):
            return NotImplemented

        return self._concrete_imatmul_(other)

    def __matmul__(self, other):
        if isinstance(other, np.ndarray) or isinstance(other, jnp.ndarray):
            return self(other)

        if not isinstance(other, LocalOperator):
            return NotImplemented

        if self == other and self.is_hermitian:
            return Squared(self)

        return self._concrete_matmul_(other)

    def _concrete_matmul_(self, other: "LocalOperator"):
        if not isinstance(other, LocalOperator):
            return NotImplemented

        op = self.copy()
        op @= other
        return op

    def _concrete_imatmul_(self, other: "LocalOperator"):
        if not isinstance(other, LocalOperator):
            return NotImplemented

        tot_operators = []
        tot_act = []
        for i in range(other._n_operators):
            act_i = other._acting_on[i, : other._acting_size[i]].tolist()
            ops, act = self._multiply_operator(other._operators[i], act_i)
            tot_operators += ops
            tot_act += act

        prod = LocalOperator(self.hilbert, tot_operators, tot_act)
        self_constant = self._constant
        if np.abs(other._constant) > self.mel_cutoff:
            self._constant = 0.0
            self *= other._constant
            self += prod
        else:
            self = prod

        if np.abs(self_constant) > self.mel_cutoff:
            self += other * self_constant

        return self

    def __truediv__(self, other):
        if not isinstance(other, numbers.Number):
            raise TypeError("Only divison by a scalar number is supported.")
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
        self._diag_mels = np.zeros((0, 0), dtype=np.complex128)
        self._mels = np.empty((0, 0, 0), dtype=np.complex128)
        self._x_prime = np.empty((0, 0, 0, 0))
        self._n_conns = np.empty((0, 0), dtype=np.intp)

        self._local_states = np.zeros((0, 0, 0), dtype=np.float64)

        self._basis = np.zeros((0, 0), dtype=np.int64)
        self._is_hermitian = True

    def _acting_on_list(self):
        acting_on = []
        for i in range(self._n_operators):
            acting_on.append(np.copy(self._acting_on[i, : self._acting_size[i]]))

        return acting_on

    def _operators_list(self):
        "A deep copy of the operators"
        operators = [np.copy(op) for op in self._operators]
        return operators

    def _add_operator(self, operator, acting_on):
        acting_on = np.asarray(acting_on, dtype=np.intp)
        operator = np.asarray(operator, dtype=np.complex128)

        if np.unique(acting_on).size != acting_on.size:
            raise ValueError("acting_on contains repeated entries.")

        if any(acting_on >= self.hilbert.size):
            raise ValueError("acting_on points to a site not in the hilbert space.")

        if operator.ndim != 2:
            raise ValueError("The operator should be a matrix")

        n_local_states_per_site = np.asarray(
            [self.hilbert.size_at_index(i) for i in acting_on]
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

            self._append_matrix(
                self._operators[support_i],
                self._diag_mels[support_i],
                self._mels[support_i],
                self._x_prime[support_i],
                self._n_conns[support_i],
                self._acting_size[support_i],
                self.mel_cutoff,
                n_local_states_per_site,
                self._local_states[support_i],
            )

            isherm = True
            for op in self._operators:
                isherm = isherm and is_hermitian(op)

            self._is_hermitian = isherm

            return

        self._n_operators += 1
        self._operators.append(operator)

        self._acting_size = np.resize(self._acting_size, self._n_operators)
        local_states_per_acting_on = [
            self.hilbert.states_at_index(i) for i in acting_on
        ]

        max_op_size = 0

        self._acting_size[-1] = acting_on.size

        self._max_op_size = max((operator.shape[0], self._max_op_size))

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

        # numpy.resize does not preserve inner content correctly,
        # we need to do this ugly thing here to prevent issues
        # hopefully this is executed only a few times in realistic situations
        if self._acting_on.shape[1] != self._max_acting_size:
            # print(f"changging acting on shape to {self._max_acting_size} from {self._acting_on.shape[1]}")
            old_n_op = self._acting_on.shape[0]

            old_array = self._acting_on
            self._acting_on = np.resize(old_array, (old_n_op, self._max_acting_size))
            self._acting_on[:, : old_array.shape[1]] = old_array

            old_array = self._diag_mels
            self._diag_mels = np.resize(old_array, (old_n_op, self._max_op_size))
            self._diag_mels[:, : old_array.shape[1]] = old_array

            old_array = self._mels
            self._mels = np.resize(
                old_array, (old_n_op, self._max_op_size, self._max_op_size - 1)
            )
            self._mels[:, : old_array.shape[1], : old_array.shape[2]] = old_array

            old_array = self._x_prime
            self._x_prime = np.resize(
                old_array,
                (
                    old_n_op,
                    self._max_op_size,
                    self._max_op_size - 1,
                    self._max_acting_size,
                ),
            )
            self._x_prime[:, :, :, :] = -1
            self._x_prime[
                :, : old_array.shape[1], : old_array.shape[2], : old_array.shape[3]
            ] = old_array

            old_array = self._n_conns
            self._n_conns = np.resize(old_array, (old_n_op, self._max_op_size))
            self._n_conns[:, : old_array.shape[1]] = old_array

            old_max_local_hilb_size = self._local_states.shape[2]
            old_array = self._local_states
            self._local_states = np.resize(
                old_array, (old_n_op, self._max_acting_size, old_max_local_hilb_size)
            )
            self._local_states[:, :, :] = np.nan
            self._local_states[:, : old_array.shape[1], :] = old_array

            old_array = self._basis
            self._basis = np.resize(old_array, (old_n_op, self._max_acting_size))
            self._basis[:, :] = 1e10
            self._basis[:, : old_array.shape[1]] = old_array

        if self._local_states.shape[2] != self._max_local_hilbert_size:

            n_op = self._acting_on.shape[0]

            old_array = self._local_states
            self._local_states = np.resize(
                old_array, (n_op, self._max_acting_size, self._max_local_hilbert_size)
            )
            self._local_states[:, :, :] = np.nan
            self._local_states[:, :, : old_array.shape[2]] = old_array

        ## add an operator to acting_on
        old_array = self._acting_on
        self._acting_on = np.resize(
            old_array, (self._n_operators, self._max_acting_size)
        )
        self._acting_on[: old_array.shape[0], : old_array.shape[1]] = old_array

        # self._acting_on[-1].fill(np.nan)
        self._acting_on[-1].fill(-1)

        acting_size = acting_on.size
        self._acting_on[-1, :acting_size] = acting_on
        if (
            self._acting_on[-1, :acting_size].max() > self.hilbert.size
            or self._acting_on[-1, :acting_size].min() < 0
        ):
            raise InvalidInputError("Operator acts on an invalid set of sites")
        ##

        ## add an operator to local_states
        old_array = self._local_states
        self._local_states = np.resize(
            old_array,
            (self._n_operators, self._max_acting_size, self._max_local_hilbert_size),
        )
        self._local_states[
            : old_array.shape[0], : old_array.shape[1], : old_array.shape[2]
        ] = old_array
        self._local_states[-1].fill(np.nan)

        for site in range(acting_size):
            self._local_states[-1, site, : n_local_states_per_site[site]] = np.asarray(
                local_states_per_acting_on[site]
            )

        ## add an operator to basis
        old_array = self._basis
        self._basis = np.resize(old_array, (self._n_operators, self._max_acting_size))
        self._basis[: old_array.shape[0], : old_array.shape[1]] = old_array

        self._basis[-1].fill(0)
        ba = 1
        for s in range(acting_on.size):
            self._basis[-1, s] = ba
            ba *= n_local_states_per_site[acting_on.size - s - 1]
        ##

        if acting_on.max() + 1 >= self._size:
            self._size = acting_on.max() + 1

        n_operators = self._n_operators
        max_op_size = self._max_op_size
        max_acting_size = self._max_acting_size

        self._diag_mels = np.resize(self._diag_mels, (n_operators, max_op_size))
        self._mels = np.resize(self._mels, (n_operators, max_op_size, max_op_size - 1))
        self._x_prime = np.resize(
            self._x_prime, (n_operators, max_op_size, max_op_size - 1, max_acting_size)
        )
        self._x_prime[-1, :, :, :] = -1
        self._n_conns = np.resize(self._n_conns, (n_operators, max_op_size))

        self._append_matrix(
            operator,
            self._diag_mels[-1],
            self._mels[-1],
            self._x_prime[-1],
            self._n_conns[-1],
            self._acting_size[-1],
            self.mel_cutoff,
            n_local_states_per_site,
            self._local_states[-1],
        )

        isherm = True
        for op in self._operators:
            isherm = isherm and is_hermitian(op)

        self._is_hermitian = isherm

    @staticmethod
    @jit(nopython=True)
    def _append_matrix(
        operator,
        diag_mels,
        mels,
        x_prime,
        n_conns,
        acting_size,
        epsilon,
        hilb_size_per_site,
        local_states_per_site,
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

        for i in range(self._n_operators):
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
                # partially intersecting support
                raise NotImplementedError(
                    "Product of intersecting LocalOperator is not implemented."
                )

        return operators, acting_on

    def copy(self):
        """Returns a copy of the operator."""
        return LocalOperator(
            hilbert=self.hilbert,
            operators=[np.copy(op) for op in self._operators],
            acting_on=self._acting_on_list(),
            constant=self._constant,
        )

    def _concrete_transpose(self):
        r"""LocalOperator: Returns the tranpose of this operator."""

        new_ops = [np.copy(ops.transpose()) for ops in self._operators]

        return LocalOperator(
            hilbert=self.hilbert,
            operators=new_ops,
            acting_on=self._acting_on_list(),
            constant=self._constant,
        )

    def transpose(self):
        return Transpose(self)

    @property
    def T(self):
        return self.transpose()

    def conjugate(self):
        r"""LocalOperator: Returns the complex conjugate of this operator."""
        new_ops = [np.copy(ops).conjugate() for ops in self._operators]

        return LocalOperator(
            hilbert=self.hilbert,
            operators=new_ops,
            acting_on=self._acting_on_list(),
            constant=np.conjugate(self._constant),
        )

    def conj(self):
        return self.conjugate()

    @property
    def H(self):
        if self.is_hermitian:
            return self

        return Adjoint(self)

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
        from a given quantum number x, it finds all other quantum numbers x' such
        that the matrix element :math:`O(x,x')` is different from zero. In general there
        will be several different connected states x' satisfying this
        condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

        This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

        Args:
            x (array): An array of shape (hilbert.size) containing the quantum numbers x.

        Returns:
            matrix: The connected states x' of shape (N_connected,hilbert.size)
            array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._get_conn_flattened_kernel(
            x.reshape((1, -1)),
            np.ones(1),
            self._local_states,
            self._basis,
            self._constant,
            self._diag_mels,
            self._n_conns,
            self._mels,
            self._x_prime,
            self._acting_on,
            self._acting_size,
        )

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
            pad,
        )

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
        pad=False,
    ):
        batch_size = x.shape[0]
        n_sites = x.shape[1]

        assert sections.shape[0] == batch_size

        n_operators = n_conns.shape[0]
        xs_n = np.empty((batch_size, n_operators), dtype=np.intp)

        tot_conn = 0
        max_conn = 0

        for b in range(batch_size):
            # diagonal element
            conn_b = 1

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

        x_prime = np.empty((tot_conn, n_sites))
        mels = np.empty(tot_conn, dtype=np.complex128)

        c = 0
        for b in range(batch_size):
            c_diag = c
            mels[c_diag] = constant
            x_batch = x[b]
            x_prime[c_diag] = np.copy(x_batch)
            c += 1
            for i in range(n_operators):

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
                            x_prime[c + cc, sites[k]] = all_x_prime[
                                i, xs_n[b, i], cc, k
                            ]
                    c += n_conn_i

            if pad:
                delta_conn = max_conn - (c - c_diag)
                mels[c : c + delta_conn].fill(0.0j)
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
        mels = np.empty(tot_conn, dtype=np.complex128)

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
        ao = []
        for actions in self._acting_on:
            _a = []
            for site in actions:
                if site >= 0:
                    _a.append(site)
            ao.append(_a)

        acting_str = f"acting_on={ao}"
        if len(acting_str) > 55:
            acting_str = f"#acting_on={len(ao)} locations"
        return f"{type(self).__name__}(dim={self.hilbert.size}, {acting_str})"
