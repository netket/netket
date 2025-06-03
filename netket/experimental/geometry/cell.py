from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp
import numpy as np
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
@dataclass(frozen=True)
class Cell:
    """A finite region in continuous space."""

    extent: tuple[float, ...]
    pbc: tuple[bool, ...]

    def __init__(
        self,
        d: int,
        L: float | Sequence[float] = 1.0,
        pbc: bool | Sequence[bool] = True,
    ) -> None:
        if d <= 0:
            raise ValueError("d must be positive")

        if hasattr(L, "__len__"):
            if len(L) != d:
                raise ValueError("Length of L must match d")
            extent = tuple(float(x) for x in L)
        else:
            extent = (float(L),) * d

        if isinstance(pbc, bool):
            pbc_tuple = (pbc,) * d
        else:
            if len(pbc) != d:
                raise ValueError("Length of pbc must match d")
            pbc_tuple = tuple(bool(x) for x in pbc)

        for e, b in zip(extent, pbc_tuple):
            if np.isinf(e) and b:
                raise ValueError(
                    "Cannot combine periodic boundary conditions with infinite extent"
                )

        object.__setattr__(self, "extent", extent)
        object.__setattr__(self, "pbc", pbc_tuple)

    def tree_flatten(self):
        data = (jnp.asarray(self.extent), jnp.asarray(self.pbc))
        return data, None

    @classmethod
    def tree_unflatten(cls, aux, data):
        extent, pbc = data
        return cls(
            d=len(extent),
            L=tuple(float(x) for x in extent),
            pbc=tuple(bool(x) for x in pbc),
        )

    @property
    def dimension(self) -> int:
        return len(self.extent)

    def distance(
        self, r1: Sequence[float] | jnp.ndarray, r2: Sequence[float] | jnp.ndarray
    ) -> jnp.ndarray:
        """Euclidean distance between ``r1`` and ``r2`` considering boundary conditions."""

        r1 = jnp.asarray(r1)
        r2 = jnp.asarray(r2)
        diff = r1 - r2
        ext = jnp.asarray(self.extent)
        pbc = jnp.asarray(self.pbc)

        shift = jnp.where(pbc, jnp.round(diff / ext) * ext, 0.0)
        diff = diff - shift
        return jnp.linalg.norm(diff, axis=-1)


@register_pytree_node_class
@dataclass(frozen=True)
class FreeSpace(Cell):
    """Infinite space without periodic boundary conditions."""

    def __init__(self, d: int) -> None:
        super().__init__(d=d, L=(np.inf,) * d, pbc=False)
