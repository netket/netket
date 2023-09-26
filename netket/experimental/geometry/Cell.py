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
import jax
import jax.numpy as jnp
import numpy as np
from . import AbstractGeometry
from typing import Optional
from netket.utils import HashableArray


def take_sub(key, x, n):
    key, subkey = jax.random.split(key)
    ind = jax.random.choice(
        subkey, jnp.arange(0, x.shape[0], 1), replace=False, shape=(n,)
    )
    return x[ind, :]


class Cell(AbstractGeometry):
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
        return True

    @property
    def volume(self) -> int:
        r"""Returns the volume of the given physical space (defined by the basis)."""
        return np.abs(np.linalg.det(self.basis))

    @property
    def extent(self):
        r"""Returns an array of the maximum extension in each spatial direction."""
        temp = self.from_basis_to_standard(jnp.ones((1, self.dim)))
        return temp

    def distance(self, x, y=None, norm=False, tri=False, mode=None):
        assert (
            self.dim == x.shape[-1]
        ), "The dimension of the geometry does not match the dimension of the positions."
        if y is None:
            dis = x[..., None, :, :] - x[..., None, :]
        else:
            assert x.shape[-1] == y.shape[-1]
            dis = x[..., None, :, :] - y[..., None, :]

        if mode == "MIC":
            dis = self.from_standard_to_basis(dis)
            dis = jnp.remainder(dis + 1.0 / 2.0, 1.0) - 1.0 / 2.0
            dis = self.from_basis_to_standard(dis)
            if tri is True:
                idx = jnp.triu_indices(dis.shape[1], 1)
                dis = dis[..., idx[0], idx[1], :]
            if norm is True:
                return dis, jnp.linalg.norm(dis)
            else:
                return dis

        elif mode == "Periodic":
            pdis = self.make_periodic(dis)

            if norm is True:
                frac = self.from_standard_to_basis(dis)
                sij = jnp.einsum("ik,kj->ij", self.basis, self.basis) / jnp.linalg.norm(
                    self.basis, axis=-1, keepdims=True
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
                if tri is True:
                    idx = jnp.triu_indices(dis.shape[1], 1)
                    pdis = pdis[..., idx[0], idx[1], :]
                    pdisnorm = t1[..., idx[0], idx[1], :] + t2[..., idx[0], idx[1], :]
                    return pdis, pdisnorm
                else:
                    return pdis, t1 + t2
            else:
                return pdis

        raise NotImplementedError

    def add(self, x, y):
        frac = self.from_standard_to_basis(x)
        frac = (frac + y) % 1.0
        return self.from_basis_to_standard(frac)

    def random_init(self, shape):
        batches, N, _ = shape
        key = jax.random.PRNGKey(42)
        key = jax.random.split(key, num=batches)

        n = int(np.ceil(N ** (1 / self.dim)))
        xs = jnp.linspace(0, 1, n)
        uniform = jnp.array(jnp.meshgrid(*(self.dim * [xs]))).T.reshape(-1, self.dim)
        uniform = jnp.tile(uniform, (batches, 1, 1))
        uniform = jax.vmap(take_sub, in_axes=(0, 0, None))(key, uniform, N)
        uniform = self.from_basis_to_standard(uniform).reshape(batches, -1)

        return uniform

    def make_periodic(self, x):
        r"""Given a batch of position vectors in the geometry, this function returns a periodic decomposition of these
        vectors.
        """
        frac = self.from_standard_to_basis(x)
        return jnp.concatenate(
            (jnp.sin(2 * jnp.pi * frac), jnp.cos(2 * jnp.pi * frac)), axis=-1
        )

    @property
    def _attrs(self):
        return (HashableArray(self.basis), HashableArray(self.volume), self.dim)

    def __repr__(self):
        return "PeriodicCell(lattice={}, volume={}, dim={})".format(
            HashableArray(self.basis), HashableArray(self.volume), self.dim
        )
