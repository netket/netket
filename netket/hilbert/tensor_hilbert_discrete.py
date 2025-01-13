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


import numpy as np
import jax.numpy as jnp

from .index import is_indexable
from .discrete_hilbert import DiscreteHilbert
from .tensor_hilbert import TensorHilbert


class TensorDiscreteHilbert(TensorHilbert, DiscreteHilbert):
    r"""Tensor product of several Discrete sub-spaces, representing the space

    In general you should not construct this object directly, but you should
    simply multiply different hilbert spaces together. In this case, Python's
    `*` operator will be interpreted as a tensor product.

    This Hilbert can be used as a replacement anywhere a Uniform Hilbert space
    is not required.

    Examples:
        Couple a bosonic mode with spins

        >>> import netket as nk
        >>> from netket.hilbert import Spin, Fock
        >>> hi = Fock(3)*Spin(0.5, 5)
        >>> print(hi)
        Fock(n_max=3, N=1)⊗Spin(s=1/2, N=5, ordering=new)
        >>> isinstance(hi, nk.hilbert.TensorHilbert)
        True
        >>> type(hi)
        <class 'netket.hilbert.tensor_hilbert_discrete.TensorDiscreteHilbert'>

    """

    def __init__(self, *hilb_spaces: DiscreteHilbert):
        r"""Constructs a tensor Hilbert space

        Args:
            *hilb: An iterable object containing at least 1 hilbert space.
        """
        if not all(isinstance(hi, DiscreteHilbert) for hi in hilb_spaces):
            raise TypeError(
                "Arguments to TensorDiscreteHilbert must all be "
                "subtypes of DiscreteHilbert. However the types are:\n\n"
                f"{list(type(hi) for hi in hilb_spaces)}\n"
            )

        shape = np.concatenate([hi.shape for hi in hilb_spaces])
        self._initialized = False
        super().__init__(hilb_spaces, shape=shape)

    @property
    def is_indexable(self) -> bool:
        """Whether the space can be indexed with an integer"""
        return all(hi.is_indexable for hi in self._hilbert_spaces) and is_indexable(
            list(hi.n_states for hi in self._hilbert_spaces)
        )

    def _setup(self):
        if not self._initialized:
            if self.is_indexable:
                self._ns_states = [hi.n_states for hi in self._hilbert_spaces]
                self._ns_states_r = np.flip(self._ns_states).tolist()
                self._cum_ns_states = np.concatenate(
                    [[0], np.cumprod(self._ns_states)]
                ).tolist()
                self._cum_ns_states_r = np.flip(
                    np.cumprod(np.concatenate([[1], np.flip(self._ns_states)]))[:-1]
                ).tolist()
                self._n_states = int(np.prod(self._ns_states))
                self._initialized = True
            else:
                raise RuntimeError("The hilbert space is too large to be indexed.")

    @property
    def is_finite(self):
        return all([hi.is_finite for hi in self._hilbert_spaces])

    @property
    def constrained(self) -> bool:
        r"""The hilbert space does not contains `prod(hilbert.shape)`
        basis states.

        Typical constraints are population constraints (such as fixed
        number of bosons, fixed magnetization...) which ensure that
        only a subset of the total unconstrained space is populated.

        Typically, objects defined in the constrained space cannot be
        converted to QuTiP or other formats.
        """
        return all([hi.constrained for hi in self._hilbert_spaces])

    def states_at_index(self, i):
        # j = self._sub_index(i)
        # return self._hilbert_spaces[j].states_at_index(i-self._cum_indices[j-1])
        return self._hilbert_spaces[self._hilbert_i[i]].states_at_index(
            i - self._delta_indices_i[i]
        )

    @property
    def n_states(self) -> int:
        self._setup()
        return self._n_states

    def _numbers_to_states(self, numbers):
        # !!! WARNING
        # This code assumes that states are stored in a MSB
        # (Most Significant Bit) format.
        # We assume that the rightmost-half indexes the LSBs
        # and the leftmost-half indexes the MSBs
        # HilbertIndex-generated states respect this, as they are:
        # 0 -> [0,0,0,0]
        # 1 -> [0,0,0,1]
        # 2 -> [0,0,1,0]
        # etc...

        self._setup()
        rem = numbers
        tmp = []
        for i, dim in enumerate(self._ns_states_r):
            rem, loc_numbers = jnp.divmod(rem, dim)
            hi_i = self._n_hilbert_spaces - (i + 1)
            tmp.append(self._hilbert_spaces[hi_i].numbers_to_states(loc_numbers))

        out = jnp.empty((numbers.size, self.size), dtype=jnp.result_type(*tmp))
        for i, dim in enumerate(self._ns_states_r):
            hi_i = self._n_hilbert_spaces - (i + 1)
            out = out.at[:, self._cum_indices[hi_i] : self._cum_sizes[hi_i]].set(tmp[i])
        return out

    def _states_to_numbers(self, states):
        # !!! WARNING
        # See note above in numbers_to_states
        self._setup()
        out = 0
        for i, dim in enumerate(self._cum_ns_states_r):
            temp = self._hilbert_spaces[i].states_to_numbers(
                states[:, self._cum_indices[i] : self._cum_sizes[i]]
            )
            out = out + temp * dim
        return out

    def states_to_local_indices(self, x):
        tmp = []
        for i, hilb_i in enumerate(self._hilbert_spaces):
            tmp.append(
                hilb_i.states_to_local_indices(
                    x[..., self._cum_indices[i] : self._cum_sizes[i]]
                )
            )
        out = jnp.empty(x.shape, dtype=jnp.result_type(*tmp))
        for i, _ in enumerate(self._hilbert_spaces):
            out = out.at[..., self._cum_indices[i] : self._cum_sizes[i]].set(tmp[i])
        return out

    def local_indices_to_states(self, x, dtype=None):
        tmp = []
        for i, hilb_i in enumerate(self._hilbert_spaces):
            tmp.append(
                hilb_i.local_indices_to_states(
                    x[..., self._cum_indices[i] : self._cum_sizes[i]]
                )
            )
        out = jnp.empty(x.shape, dtype=jnp.result_type(*tmp))
        for i, _ in enumerate(self._hilbert_spaces):
            out = out.at[..., self._cum_indices[i] : self._cum_sizes[i]].set(tmp[i])
        return out
