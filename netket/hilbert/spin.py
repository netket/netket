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

from typing import Optional
from fractions import Fraction


import numpy as np

from netket.utils import StaticRange

from .homogeneous import HomogeneousHilbert
from .constraint import DiscreteHilbertConstraint, SumConstraint


def _check_total_sz(total_sz, S, size):
    if total_sz is None:
        return

    local_size = 2 * S + 1

    m = round(2 * total_sz)
    if np.abs(m) > size * (2 * S):
        raise ValueError(
            "Cannot fix the total magnetization: 2|M| cannot " "exceed Nspins."
        )

    # If half-integer spins (1/2, 3/2)
    if local_size % 2 == 0:
        # Check that the total magnetization is odd if odd spins or even if even
        # number of spins
        if (size + m) % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization: Nspins + 2*totalSz must be even."
            )
    # else if full-integer (S=1,2)
    else:
        if m % 2 != 0:
            raise ValueError(
                "Cannot fix the total magnetization to a half-integer number"
            )


class Spin(HomogeneousHilbert):
    r"""Hilbert space obtained as tensor product of local spin states.

    The correspondance between basis elements of the hilbert space and the configurations
    that are then fed to variational states is as follows:

    +----------+----------------------------------+-----------------------------------+
    | State    | Old Behavior                     | New Behavior                      |
    |          | :code:`inverted_ordering=True`   | :code:`inverted_ordering=False`   |
    +==========+==================================+===================================+
    | ↑ ↑ ↑    | -1 -1 -1                         | +1 +1 +1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↑ ↑ ↓    | -1 -1 +1                         | +1 +1 -1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↑ ↓ ↑    | -1 +1 -1                         | +1 -1 +1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↑ ↓ ↓    | -1 +1 +1                         | +1 -1 -1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↓ ↑ ↑    | +1 -1 -1                         | -1 +1 +1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↓ ↑ ↓    | +1 -1 +1                         | -1 +1 -1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↓ ↓ ↑    | +1 +1 -1                         | -1 -1 +1                          |
    +----------+----------------------------------+-----------------------------------+
    | ↓ ↓ ↓    | +1 +1 +1                         | -1 -1 -1                          |
    +----------+----------------------------------+-----------------------------------+

    The old behaviour is the default behaviour of NetKet 3.14 and before, while the new
    behaviour will become the default starting 1st january 2025.
    For that reason, in the transition period, we will print warnings asking to explicitly
    specify which ordering you want

    .. warning::

        The ordering of the Spin Hilbert space basis has historically always been
        such that `-1=↑, 1=↓`, but it will be changed 1st january 2025 to
        be such that `1=↑, -1=↓`.

        The change will break:
            - code that relies on the assumption that -1=↑;
            - all saves because the inputs to the network will change;
            - custom operators that rely on the basis being ordered;

        To avoid distruption, NetKet will support **both** conventions in the (near)
        future. You can specify the ordering you need with :code:`inverted_ordering = True`
        (historical ordering) or :code:`inverted_ordering=False` (future default behaviour).

        If you do not specify this flag, a future version of NetKet might break your
        serialized weights or other logic, so we strongly reccomend that you either
        limit yourself to NetKet 3.14, or that you specify :code:`inverted_ordering`
        explicitly.

    """

    def __init__(
        self,
        s: float,
        N: int = 1,
        *,
        total_sz: float | None = None,
        constraint: DiscreteHilbertConstraint | None = None,
        inverted_ordering: bool = False,
    ):
        r"""Hilbert space obtained as tensor product of local spin states.

        .. note::

            Since NetKet 3.16 (January 2025) the default ordering of the Spin Hilbert space
            basis has changed. The new default is such that `1=↑, -1=↓`. This change can
            be controlled by the `inverted_ordering` flag. If you do not specify this flag,
            you will get the new behaviour.

            To ensure that the old behaviour is maintained, you should specify
            `inverted_ordering=True`. If you want to opt into the new default
            you should specify `inverted_ordering=False`.

        Args:
            s: Spin at each site. Must be integer or half-integer.
            N: Number of sites (default=1)
            total_sz: If given, constrains the total spin of system to a particular
                value.
            constraint: A custom constraint on the allowed configurations. This argument
                cannot be specified at the same time as :code:`total_sz`. The constraint
                must be a subclass of :class:`~netket.hilbert.DiscreteHilbertConstraint`.
            inverted_ordering: Flag to specify the ordering of the Local basis. Historically
                NetKet has always used the convention `-1=↑, 1=↓` (corresponding to
                :code:`inverted_ordering=True`, but we will change it to `1=↑, -1=↓` (
                :code:`inverted_ordering=False`).
                The default as of September 2024 (NetKet 3.14) is :code:`inverted_ordering=True`, but
                we will change it in the near future.
                The change will (i) break code that relies on the assumption that -1=↑, and
                (ii) will break all saves because the inputs to the network will change.


        Examples:
           Simple spin hilbert space.

           >>> import netket as nk
           >>> hi = nk.hilbert.Spin(s=1/2, N=4)
           >>> print(hi.size)
           4
        """
        local_size = round(2 * s + 1)
        assert int(2 * s + 1) == local_size

        # TODO: Remove in NetKet 3.17
        if inverted_ordering is None:
            inverted_ordering = False

        if not inverted_ordering:
            # Reasonable, new ordering where  1=↑ -1=↓
            local_states = StaticRange(
                local_size - 1,  # type: ignore[arg-type]
                -2,  # type: ignore[arg-type]
                local_size,
            )
        else:
            # Old ordering where -1=↑ 1=↓
            local_states = StaticRange(
                1 - local_size,  # type: ignore[arg-type]
                2,  # type: ignore[arg-type]
                local_size,
            )

        _check_total_sz(total_sz, s, N)
        if total_sz is not None:
            if constraint is not None:
                raise ValueError(
                    "Cannot specify at the same time a total magnetization "
                    "constraint and a `custom_constraint."
                )
            constraint = SumConstraint(round(2 * total_sz))

        self._total_sz = total_sz
        self._s = s
        self._inverted_ordering = inverted_ordering

        super().__init__(local_states, N, constraint=constraint)

    def __pow__(self, n):
        if not self.constrained:
            return Spin(self._s, self.size * n)

        return NotImplemented

    def _mul_sametype_(self, other):
        assert type(self) == type(other)
        if self._s == other._s:
            if not self.constrained and not other.constrained:
                return Spin(s=self._s, N=self.size + other.size)

        return NotImplemented

    def ptrace(self, sites: int | list) -> Optional["Spin"]:
        if isinstance(sites, int):
            sites = [sites]

        for site in sites:
            if site < 0 or site >= self.size:
                raise ValueError(
                    f"Site {site} not in this hilbert space of site {self.size}"
                )

        if self.constrained:
            raise TypeError("Cannot take the partial trace with a constraint.")

        Nsites = len(sites)

        if self.size - Nsites == 0:
            return None
        else:
            return Spin(s=self._s, N=self.size - Nsites)

    def __repr__(self):
        if self._total_sz is not None:
            constraint = f", total_sz={self._total_sz}"
        elif self.constrained:
            constraint = f", {self._constraint}"
        else:
            constraint = ""
        ordering = "inverted" if self._inverted_ordering else "new"
        return f"Spin(s={Fraction(self._s)}, N={self.size}, ordering={ordering}{constraint})"

    @property
    def _attrs(self):
        return (self.size, self._s, self._inverted_ordering, self.constraint)
