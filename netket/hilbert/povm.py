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

import abc

import jax.numpy as jnp

import numpy as np

from netket.utils import StaticRange
from .homogeneous import HomogeneousHilbert


class AbstractPOVM(HomogeneousHilbert):
    """Computational basis for a POVM."""

    def __init__(self, N: int = 1):
        """Initializes a new instance of the POVM class."""
        self._M = self._compute_M()
        self._T = jnp.einsum("aij,bji -> ab", self._M, self._M)
        self._Tinv = jnp.linalg.inv(self.T)

        local_states = StaticRange(0, 1, len(self.M))
        super().__init__(local_states, N)

    @abc.abstractmethod
    def _compute_M(self):
        raise NotImplementedError()

    @property
    def M(self):
        return self._M

    @property
    def T(self):
        return self._T

    @property
    def Tinv(self):
        return self._Tinv

    def __repr__(self):
        return f"{type(self).__name__}(N={self.size})"


class POVMTethra(AbstractPOVM):
    def __init__(self, N: int = 1, theta: float = 0.0, phi: float = 0.0):
        self._theta = theta
        self._phi = phi
        super().__init__(N)

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @property
    def _attrs(self):
        return (self.size, self.theta, self._phi)

    def _compute_M(self):
        """Returns 4 POVM measurement operators.

        Args:
            * ``theta``: angle theta on the Bloch - sphere
            * ``phi``: angle phi on the Bloch - sphere
            * ``name``: specifier of the POVM

        Returns:
            jnp.array with the leading axis giving the different POVM-Measurement operators.

        """

        s = jnp.array(
            [
                [0.0, 0.0, 1.0],
                [2.0 * jnp.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0],
                [-jnp.sqrt(2.0) / 3.0, jnp.sqrt(2.0 / 3.0), -1.0 / 3.0],
                [-jnp.sqrt(2.0) / 3.0, -jnp.sqrt(2.0 / 3.0), -1.0 / 3.0],
            ]
        )

        # rotate s
        t_cos = np.cos(self.theta)
        t_sin = jnp.sin(self.theta)
        p_cos = jnp.cos(self.phi)
        p_sin = jnp.sin(self.phi)

        rotator_theta = jnp.array(
            [[t_cos, 0.0, t_sin], [0.0, 1.0, 0.0], [-t_sin, 0.0, t_cos]]
        )
        rotator_phi = jnp.array(
            [[p_cos, -p_sin, 0.0], [p_sin, p_cos, 0.0], [0.0, 0.0, 1.0]]
        )
        rotator = jnp.dot(jnp.transpose(rotator_theta), jnp.transpose(rotator_phi))

        s = jnp.dot(s, rotator)
        M = (
            jnp.array(
                jnp.eye(2) + jnp.einsum("ak, kij -> aij", s, get_paulis()),
                dtype=complex,
            )
            / 4
        )
        return M


def get_paulis():
    """
    Returns the Pauli matrices.
    """
    return jnp.array(
        [[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1.0j], [1.0j, 0.0]], [[1, 0], [0, -1]]]
    )
