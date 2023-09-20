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
import jax.numpy as jnp
from .continuous_geometry import Geometry
from typing import Optional, Union, Tuple
from netket.utils import HashableArray


class Cell(Geometry):
    def __init__(
        self, lattice: HashableArray, L: Optional[Union[float, Tuple[float]]] = None
    ):
        self._L = L
        super().__init__(pbc=True, lattice=jnp.array(lattice))

    @property
    def volume(self) -> int:
        r"""The volume of the Hilbert space if defined"""
        return jnp.abs(jnp.linalg.det(self._lat))

    def distance(self, x, y=None, norm=False, mode=None) -> list[list]:
        r"""x = (..., N, dim)"""
        if y is None:
            dis = (x[..., None, :, :] - x[..., None, :]).reshape(
                -1, x.shape[1], x.shape[1], self.dim
            )
        else:
            assert x.shape[-1] == y.shape[-1]
            dis = (x[..., None, :, :] - y[..., None, :]).reshape(
                -1, x.shape[1], y.shape[1], self.dim
            )

        if mode == "minimal" and self._L is None:
            raise ValueError(
                """Minimum Image Convention is only implemented for square boxes."""
            )
        elif mode == "minimal" and self._L is not None:
            dis = jnp.remainder(dis + self._L / 2.0, self._L) - self._L / 2.0
            return dis

        elif mode == "periodic" and norm is True:
            frac = self.from_standard_to_lat(dis)
            sij = jnp.einsum("ik,kj->ij", self.lattice, self.lattice) / jnp.linalg.norm(
                self.lattice, axis=-1, keepdims=True
            )
            t1 = jnp.einsum(
                "...i,ij,...j->...",
                1 - jnp.cos(2 * jnp.pi * frac),
                sij,
                1 - jnp.cos(2 * jnp.pi * frac),
            )
            t2 = jnp.einsum(
                "...i,ij,...j->...",
                jnp.sin(2 * jnp.pi * frac),
                sij,
                jnp.sin(2 * jnp.pi * frac),
            )
            return self.make_periodic(dis), t1 + t2

        elif mode == "periodic":
            return self.make_periodic(dis)

    def make_periodic(self, x):
        frac = self.from_standard_to_lat(x)
        return jnp.concatenate(
            (jnp.sin(2 * jnp.pi * frac), jnp.cos(2 * jnp.pi * frac)), axis=-1
        )

    def __repr__(self):
        return "PeriodicCell(lattice={}, volume={}, dim={})".format(
            self.lattice, self.volume, self.dim
        )
