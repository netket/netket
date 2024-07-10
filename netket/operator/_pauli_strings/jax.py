# Copyright 2023 The NetKet Authors - All rights reserved.
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

from functools import partial, wraps
from typing import Optional, Union, TYPE_CHECKING

import numpy as np

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import AbstractHilbert, HomogeneousHilbert
from netket.errors import concrete_or_error, JaxOperatorSetupDuringTracingError
from netket.utils.types import DType
from netket.utils import HashableArray

from .._discrete_operator_jax import DiscreteJaxOperator

from .base import PauliStringsBase

if TYPE_CHECKING:
    from .numba import PauliStrings

# pauli-strings operator written in nJax
# the general idea is the following:

# observe that Y = -i Z X
# therefore for every pauli in a given operator we
# (1.) apply X  (flip site)                  -- if it is X or Y
# (2.) absorb -i to the weights              -- if it is Y
# (3.) apply Z (pick up -1. if site is up/1) -- if it is Y or Z


# TODO special case for the diagonal
# TODO eventually also implement _ising_conn_states_jax with indexing instead of mask
# TODO eventually add version with sparse jax arrays (achieving the same as indexing)


# duplicated from ising
def _ising_conn_states_jax(x, cond, local_states):
    was_state_0 = x == local_states[0]
    state_0 = jnp.asarray(local_states[0], dtype=x.dtype)
    state_1 = jnp.asarray(local_states[1], dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, state_0, state_1)


def pack_internals(operators, weights, cutoff=0):
    # here we group together operators with same final state
    #
    # The final state is determined by the sites we flip,
    # we store this in the keys of `acting`
    #
    # Each value of `acting` contains a list, with an entry for of the opeartors
    # which does the same flips (given by the key). Each entry contains
    # 1. a weight
    # 2. list of sites we apply the Z on (those are the sites we
    #    will need to check in the operator to determine if the sign is flipped or not)

    acting = {}

    def find_char(s, ch):
        return [i for i, ltr in enumerate(s) if ltr == ch]

    def append(key, k):
        # convert list to tuple
        key = tuple(sorted(key))  # order of X and Y does not matter
        if key in acting:
            acting[key].append(k)
        else:
            acting[key] = [k]

    for i, op in enumerate(operators):
        b_to_change = []  # list of all the sites we will need to act on with X
        b_z_check = []  # list of all the sites we act on with Z
        b_weight = weights[i]

        if abs(b_weight) <= cutoff:
            continue

        x_ops = find_char(op, "X")  # find sites we act on with X w/ op
        if len(x_ops):
            b_to_change += x_ops

        y_ops = find_char(op, "Y")  # find sites we act on with Y w/ op
        if len(y_ops):
            b_to_change += y_ops
            b_weight *= (-1.0j) ** (len(y_ops))  # absorb the -i into weights
            b_z_check += y_ops

        z_ops = find_char(op, "Z")  # find sites we act on with Z w/ op
        if len(z_ops):
            b_z_check += z_ops

        # by appending we concat all (b_weights, b_z_check) which have the same b_to_change
        # i.e. the one which we act on with X on the same sites (also X coming from Y obviously)

        # sort b_z_check in ascending order, for better locality
        b_z_check = list(sorted(b_z_check))

        # If there is an even number of Y in a string, the weight should be real
        if np.isreal(b_weight):
            b_weight = b_weight.real

        append(b_to_change, (b_weight, b_z_check))
    return acting


def sites_to_mask(sites, n_sites, dtype=bool):
    mask = np.zeros(n_sites, dtype=dtype)
    mask[(sites,)] = 1
    return mask


_available_modes = ["index", "mask"]


def _check_mode(mode):
    if mode not in _available_modes:
        raise ValueError(
            f"unknown mode {mode}. Available modes are {_available_modes}."
        )


def pack_internals_jax(
    operators,
    weights,
    mask_dtype=jnp.bool_,
    index_dtype=jnp.int32,
    weight_dtype=None,
    mode="mask",
):
    """
    Take the internal lazy representation of a paulistrings operator and returns the
    arrays needed for the jax implementation.

    This takes as input a numpy array of strings (`operators`) of equal lengths
    containing only the characters 'X', 'Y', 'Z' or 'I',  and an equal length
    vector of real or complex coefficients (`weights`).

    `weights` must be a numpy-array with statically known values otherwise an error
    is thrown.

    Returns a dictionary with all the data fields
    """
    # index_dtype needs to be signed (we use -1 for padding)

    _check_mode(mode)

    # group together operators with same final state (i.e. those which flip the same sites)
    # see code of pack_internals for more details
    acting = pack_internals(operators, weights)

    n_sites = len(operators[0])

    # now group those ones which have the same number of operators for a given final state
    acting_by_num_ops = {}
    for k, v in acting.items():
        num = len(v)
        acting_by_num_ops[num] = acting_by_num_ops.get(num, []) + [(k, v)]

    x_flip_masks = {}
    weights = {}
    z_sign_masks = {}
    z_sign_indices = {}
    z_sign_indices_masks = {}

    for l, ops in acting_by_num_ops.items():
        x_flip_masks_l = []
        weights_l = []
        z_sign_masks_l = []  # if we decide to use masks
        z_sign_indices_l = []  # if we decide to use indexing

        for sites_to_flip, rest in ops:
            w = [r[0] for r in rest]
            sites_for_sgn = [r[1] for r in rest]
            x_flip_mask = sites_to_mask(sites_to_flip, n_sites)
            z_sign_mask = list(
                map(partial(sites_to_mask, n_sites=n_sites), sites_for_sgn)
            )

            x_flip_masks_l.append(x_flip_mask)
            weights_l.append(w)
            z_sign_masks_l.append(z_sign_mask)
            z_sign_indices_l.append(sites_for_sgn)

        # turn into arrays
        x_flip_masks[l] = jnp.array(x_flip_masks_l, dtype=mask_dtype)
        if weight_dtype is not None:
            weights[l] = jnp.array(weights_l, dtype=weight_dtype)
        else:
            weights[l] = jnp.array(weights_l)

        z_sign_masks[l] = jnp.array(z_sign_masks_l, dtype=mask_dtype)

        # prepare index arrays if we are indexing
        num_z = [len(y) for x in z_sign_indices_l for y in x]
        maxlen = max(num_z, default=0)
        pad = maxlen != min(num_z, default=0)
        if pad:
            # arbitrarily fill with -1
            tmp1 = np.full((len(z_sign_indices_l), l, maxlen), -1, dtype=index_dtype)
            tmp2 = np.full((len(z_sign_indices_l), l, maxlen), False, dtype=mask_dtype)
            for i, inds in enumerate(z_sign_indices_l):
                for j, ind in enumerate(inds):
                    tmp1[i, j, : len(ind)] = ind
                    tmp2[i, j, : len(ind)] = True
            z_sign_indices[l] = jnp.array(tmp1)
            z_sign_indices_masks[l] = jnp.array(tmp2)
        else:  # no padding needed, all have the same length
            z_sign_indices[l] = jnp.array(z_sign_indices_l, dtype=index_dtype)
            z_sign_indices_masks[l] = None

    # transform the arrays into lists so that we have consistent ordering
    # as we will concatenate the results in the operator
    keys = sorted(x_flip_masks.keys())
    x_flip_masks = [x_flip_masks[k] for k in keys]
    weights = [weights[k] for k in keys]

    # TODO here would be the place we could decide wether to use index or mask
    # depending on how much we padded, use a hybrid scheme etc
    z_sign_masks = [z_sign_masks[k] if (mode == "mask") else None for k in keys]
    z_sign_indices = [z_sign_indices[k] if (mode == "index") else None for k in keys]
    z_sign_indices_masks = [
        z_sign_indices_masks[k] if (mode == "index") else None for k in keys
    ]

    x_flip_masks_stacked = jnp.concatenate(x_flip_masks, axis=0)
    z_data = (weights, z_sign_masks, z_sign_indices, z_sign_indices_masks)
    return x_flip_masks_stacked, z_data


@jax.jit
def _pauli_strings_mels_jax(local_states, z_data, x):
    # supports both masks and indexing (can be padded, so also with a mask but smaller)
    # which path is taken is flexible, and can be fully determined by z_data
    # z_data: a list of tuples weights, z_sign_mask, z_sign_indices, z_sign_indexmask
    state1 = local_states[1]
    was_state_1 = x == state1
    mels = []
    for w, z_sign_mask, z_sign_indices, z_sign_indexmask in zip(*z_data):
        if z_sign_mask is not None:  # use masks
            was_state = was_state_1[..., None, None, :]
            mask = z_sign_mask
        else:  # use indexing
            assert z_sign_indices is not None
            was_state = x[..., z_sign_indices] == state1
            if z_sign_indexmask is None:  # no padding, so no mask necessary
                mask = 1
            else:
                mask = z_sign_indexmask
        # sgn = (-1)**(was_state*mask).sum(axis=-1)
        sgn = (1 - 2 * (was_state * mask).astype(np.int8)).prod(
            axis=-1, promote_integers=False
        )
        mels.append(jnp.einsum("...ab,ab->...a", sgn, w))
    return jnp.concatenate(mels, axis=-1)


@jax.jit
def _pauli_strings_kernel_jax(local_states, x_flip_masks_all, z_data, x, cutoff=None):
    mels = _pauli_strings_mels_jax(local_states, z_data, x)
    if cutoff is not None:
        nonzero_mels_mask = jnp.abs(mels) > cutoff
        mels = jax.lax.select(nonzero_mels_mask, mels, jnp.zeros_like(mels))
        # only flip if corresponding mel is nonzero
        x_flip_masks_all = x_flip_masks_all & nonzero_mels_mask[..., None]
    else:
        nonzero_mels_mask = None
    # can re-use function from Ising
    x_prime = _ising_conn_states_jax(x[..., None, :], x_flip_masks_all, local_states)
    # TODO we could optionally move nonzeros to front here
    return x_prime, mels, nonzero_mels_mask


@jax.jit
def _pauli_strings_n_conn_jax(local_states, x_flip_masks_all, z_data, x, cutoff=None):
    _, _, nonzero_mels_mask = _pauli_strings_kernel_jax(
        local_states, x_flip_masks_all, z_data, x, cutoff
    )
    if nonzero_mels_mask is not None:
        return nonzero_mels_mask.sum(axis=-1, dtype=np.int32)
    else:
        max_conn_size = x_flip_masks_all.shape[0]
        return jnp.full(x.shape[:-1], max_conn_size, dtype=np.int32)


@register_pytree_node_class
class PauliStringsJax(PauliStringsBase, DiscreteJaxOperator):
    """
    Jax-compatible version of :class:`netket.operator.PauliStrings`.
    """

    @wraps(PauliStringsBase.__init__)
    def __init__(
        self,
        hilbert: AbstractHilbert,
        operators: Union[None, str, list[str]] = None,
        weights: Union[None, float, complex, list[Union[float, complex]]] = None,
        *,
        cutoff: float = 1.0e-10,
        dtype: Optional[DType] = None,
        _mode: str = "index",
    ):
        super().__init__(hilbert, operators, weights, cutoff=cutoff, dtype=dtype)

        if len(self.hilbert.local_states) != 2:
            raise ValueError(
                "PauliStringsJax only supports Hamiltonians with two local states"
            )

        # check that it is homogeneous, throw error if it's not
        if not isinstance(self.hilbert, HomogeneousHilbert):
            local_states = self.hilbert.states_at_index(0)
            if not all(
                np.allclose(local_states, self.hilbert.states_at_index(i))
                for i in range(self.hilbert.size)
            ):
                raise ValueError(
                    "Hilbert spaces with non homogeneous local_states are not "
                    "yet supported by PauliStrings."
                )

        # private variable for setting the mode
        # currently there are two modes:
        # index: indexes into the vector to flip qubits and compute the sign
        #           faster if the strings act only on a few qubits
        # mask: uses masks to flip qubits and compute the sign
        #          faster if the strings act on many of the qubits
        #          (and possibly on gpu)
        # By adapting pack_internals_jax hybrid approaches are also possible.
        # depending on performance tests we might expose or remove it
        self._hi_local_states = tuple(self.hilbert.local_states)
        self._initialized = False
        self._mode = _mode

    @property
    def _mode(self):
        """
        (Internal) Indexing mode of the operator.

        Valid values are "index" or "mask".

        'Index' uses the standard LocalOperator-like indexing of changed points,
        while the latter uses constant-size masks.

        The latter does not really need recompilation for paulistrings with
        different values, and this could be changed in the future.
        """
        return self._mode_attr

    @_mode.setter
    def _mode(self, mode):
        _check_mode(mode)
        self._mode_attr = mode
        self._reset_caches()

    @property
    def max_conn_size(self) -> int:
        """The maximum number of non zero ⟨x|O|x'⟩ for every x."""
        self._setup()
        return self._x_flip_masks_stacked.shape[0]

    def _setup(self, force=False):
        if force or not self._initialized:
            weights = concrete_or_error(
                np.asarray,
                self.weights,
                JaxOperatorSetupDuringTracingError,
                self,
            )

            # Necessary for the tree_flatten in jax.jit, because
            # metadata must be hashable and comparable. We don't
            # want to re-hash it at every unpacking so we do it
            # once in here.
            if self._mode == "index":
                self._operators_hashable = HashableArray(self.operators)
            else:
                self._operators_hashable = None

            x_flip_masks_stacked, z_data = pack_internals_jax(
                self.operators, weights, weight_dtype=self.dtype, mode=self._mode
            )
            self._x_flip_masks_stacked = x_flip_masks_stacked
            self._z_data = z_data
            self._initialized = True

    def _reset_caches(self):
        super()._reset_caches()
        self._initialized = False

    def n_conn(self, x):
        self._setup()
        return _pauli_strings_n_conn_jax(
            self._hi_local_states,
            self._x_flip_masks_stacked,
            self._z_data,
            x,
            cutoff=self._cutoff,
        )

    def get_conn_padded(self, x):
        self._setup()
        xp, mels, _ = _pauli_strings_kernel_jax(
            self._hi_local_states,
            self._x_flip_masks_stacked,
            self._z_data,
            x,
            cutoff=self._cutoff,
        )
        return xp, mels

    def tree_flatten(self):
        self._setup()
        data = (self.weights, self._x_flip_masks_stacked, self._z_data)
        metadata = {
            "hilbert": self.hilbert,
            "operators": self._operators_hashable,
            "dtype": self.dtype,
            "mode": self._mode,
        }
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        (weights, xm, zd) = data
        hi = metadata["hilbert"]
        operators_hashable = metadata["operators"]
        dtype = metadata["dtype"]
        mode = metadata["mode"]

        op = cls(hi, dtype=dtype, _mode=mode)
        op._operators = (
            operators_hashable.wrapped if operators_hashable is not None else None
        )
        op._operators_hashable = operators_hashable
        op._weights = weights
        op._x_flip_masks_stacked = xm
        op._z_data = zd
        op._initialized = True
        return op

    def to_numba_operator(self) -> "PauliStrings":  # noqa: F821
        """
        Returns the standard numba version of this operator, which is an
        instance of :class:`netket.operator.PauliStrings`.
        """
        from .numba import PauliStrings

        return PauliStrings(
            self.hilbert,
            self.operators,
            self.weights,
            dtype=self.dtype,
            cutoff=self._cutoff,
        )
