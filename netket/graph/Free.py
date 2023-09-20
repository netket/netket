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


class Free(Geometry):
    def __init__(
        self,
        dim: float,
    ):
        super().__init__(pbc=False, dim=dim)

    @property
    def volume(self) -> int:
        r"""The volume of the Hilbert space if defined"""
        raise ValueError("""The volume is not defined for free space.""")

    def distance(self, x, y=None, norm=False, mode=None) -> list[list]:
        if y is None:
            dis = x[..., None, :, :] - x[..., None, :]
        else:
            assert x.shape[-1] == y.shape[-1]
            dis = x[..., None] - y[..., None, :]
        if mode is not None:
            raise ValueError(
                """There is no special mode for free space distance computation."""
            )
        if norm and y is None:
            return dis, jnp.linalg.norm(dis + jnp.eye(dis.shape[1]), axis=-1) * (
                1 - jnp.eye(dis.shape[1])
            )
        if norm and y is not None:
            raise ValueError("""Why do you want to take a safe norm?""")
        return dis

    def __repr__(self):
        return "FreeSpace(dim={})".format(self.dim)
