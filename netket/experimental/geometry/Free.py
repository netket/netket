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

from typing import Optional
from . import AbstractGeometry


class Free(AbstractGeometry):
    def __init__(
        self,
        dim: Optional[int] = None,
        basis: Optional = None,
    ):
        """
        Construct a periodic geometry in continuous space, given the dimension or a basis. If only the dimension is
        given the standard basis is assumed, e.g. (1,0), (0,1) for 2D space.

        Args:
            dim: (Optional) The number of spatial dimensions of the physical space. If None, a basis has to be
                            specified. If int and basis is None, the standard basis is assumed.
            basis: (Optional) A basis for the physical space. If basis is None, the standard basis is assumed.
        """
        if dim is None and basis is None:
            raise ValueError(
                """Specify either the dimension of the geometry or provide a basis for it."""
            )

        if basis is None:
            basis = jnp.eye(dim)

        super().__init__(basis=basis)

    @property
    def pbc(self):
        return False

    def distance(self, x, y=None, norm=False, tri=False, mode="Euclidean"):
        assert (
            self.dim == x.shape[-1]
        ), "The dimension of the geometry does not match the dimension of the positions."
        if mode != "Euclidean":
            raise ValueError(
                """There is only the Euclidean mode for free space distance computation."""
            )
        if mode == "Euclidean":
            if y is None:
                dis = x[..., None, :, :] - x[..., None, :]
            else:
                assert x.shape[-1] == y.shape[-1]
                dis = x[..., None, :, :] - y[..., None, :]

            if tri:
                idx = jnp.triu_indices(dis.shape[1], 1)
                dis = dis[..., idx[0], idx[1], :]

            if norm and y is None:
                return dis, jnp.linalg.norm(dis + jnp.eye(dis.shape[1]), axis=-1) * (
                    1 - jnp.eye(dis.shape[1])
                )
            if norm and y is not None:
                dis, jnp.linalg.norm(dis, axis=-1)

            return dis

        raise NotImplementedError

    def add(self, x, y):
        return x + y

    def random_init(self, shape):
        batches, N, _ = shape
        return jnp.zeros((shape[0], N * self.dim))

    @property
    def _attrs(self):
        return (self.dim,)

    def __repr__(self):
        return "FreeSpace(dim={})".format(self.dim)
