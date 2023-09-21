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
        self, lattice: Optional = None, L: Optional[Union[float, Tuple[float]]] = None
    ):
        if lattice is None and L is None:
            raise ValueError("Either provide a lattice or a boxsize.")
        if lattice is None:
            if isinstance(L, float):
                # assume 1D (cubic) box
                lattice = jnp.array([L])
            elif isinstance(L, tuple):
                lattice = jnp.array(L) * jnp.eye(len(L))
            else:
                raise ValueError("L must be a scalar or a tuple.")

        super().__init__(pbc=True, lattice=jnp.array(lattice))

    @property
    def volume(self) -> int:
        return jnp.abs(jnp.linalg.det(self._lat))

    @property
    def extent(self):
        temp = self.from_lat_to_standard(jnp.ones((1, self.dim)))
        return temp

    def distance(self, x, y=None, norm=False, tri=False, mode=None) -> list[list]:
        if y is None:
            dis = x[..., None, :, :] - x[..., None, :]
        else:
            assert x.shape[-1] == y.shape[-1]
            dis = x[..., None, :, :] - y[..., None, :]

        if mode == "minimal":
            dis = self.from_standard_to_lat(dis)
            dis = jnp.remainder(dis + 1.0 / 2.0, 1.0) - 1.0 / 2.0
            dis = self.from_lat_to_standard(dis)
            if tri is True:
                idx = jnp.triu_indices(dis.shape[1], 1)
                dis = dis[..., idx[0], idx[1], :]
            if norm is True:
                return dis, jnp.linalg.norm(dis)
            else:
                return dis

        elif mode == "periodic":
            pdis = self.make_periodic(dis)

            if norm is True:
                frac = self.from_standard_to_lat(dis)
                sij = jnp.einsum(
                    "ik,kj->ij", self.lattice, self.lattice
                ) / jnp.linalg.norm(self.lattice, axis=-1, keepdims=True)
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
                if tri is True:
                    idx = jnp.triu_indices(dis.shape[1], 1)
                    pdis = pdis[..., idx[0], idx[1], :]
                    pdisnorm = t1[..., idx[0], idx[1], :] + t2[..., idx[0], idx[1], :]
                    return pdis, pdisnorm
                else:
                    return pdis, t1 + t2
            else:
                return pdis

    def back_to_box(self, x, delta):
        frac = self.from_standard_to_lat(x)
        frac = (frac + delta) % 1.0
        return self.from_lat_to_standard(frac)

    def make_periodic(self, x):
        frac = self.from_standard_to_lat(x)
        return jnp.concatenate(
            (jnp.sin(2 * jnp.pi * frac), jnp.cos(2 * jnp.pi * frac)), axis=-1
        )

    @property
    def _attrs(self):
        return (HashableArray(self.lattice), HashableArray(self.volume), self.dim)

    def __repr__(self):
        return "PeriodicCell(lattice={}, volume={}, dim={})".format(
            HashableArray(self.lattice), HashableArray(self.volume), self.dim
        )
