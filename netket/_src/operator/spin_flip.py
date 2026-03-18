from math import comb

import numpy as np

import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.hilbert import Spin, SpinOrbitalFermions
from netket.hilbert.constraint import SumConstraint
from netket.jax import canonicalize_dtypes
from netket.operator import DiscreteJaxOperator


def _spin_hilbert_supports_spin_flip(hilbert: Spin) -> bool:
    return (not hilbert.constrained) or (
        isinstance(hilbert.constraint, SumConstraint)
        and hilbert.constraint.sum_value == 0
    )


@register_pytree_node_class
class SpinFlipOperator(DiscreteJaxOperator):
    def __init__(self, hilbert: Spin | SpinOrbitalFermions, *, dtype=float):
        if isinstance(hilbert, Spin):
            if not _spin_hilbert_supports_spin_flip(hilbert):
                raise ValueError(
                    "Spin flip is only supported on unconstrained `Spin` Hilbert spaces "
                    "or on `Spin` Hilbert spaces constrained to zero total magnetization."
                )
        elif isinstance(hilbert, SpinOrbitalFermions):
            if hilbert.n_spin_subsectors == 1:
                raise ValueError(
                    "Spin flip is only defined for spinful `SpinOrbitalFermions` "
                    "Hilbert spaces."
                )
            if hilbert.n_spin_subsectors != 2:
                raise ValueError(
                    "Spin flip requires exactly 2 spin subsectors, "
                    f"but got {hilbert.n_spin_subsectors}."
                )
            n_fps = hilbert.n_fermions_per_spin
            if n_fps[0] is None or n_fps[1] is None:
                raise ValueError(
                    "Spin flip requires fixed fermion numbers per spin sector "
                    "(specify `n_fermions_per_spin`)."
                )
            if n_fps[0] != n_fps[1]:
                raise ValueError(
                    "Spin flip does not preserve the spin-subsector constraints: "
                    f"n_fermions_per_spin={n_fps} requires equal up and down counts."
                )
        else:
            raise TypeError(
                "Spin flip is only defined on `netket.hilbert.Spin` and "
                "`netket.hilbert.SpinOrbitalFermions`."
            )

        super().__init__(hilbert)
        self._dtype = canonicalize_dtypes(dtype=dtype)

    def tree_flatten(self):
        return (), {"hilbert": self.hilbert, "dtype": self.dtype}

    @classmethod
    def tree_unflatten(cls, struct_data, array_data):
        return cls(**struct_data)

    def __repr__(self):
        return f"SpinFlipOperator({self.hilbert}, dtype={self.dtype})"

    def __hash__(self):
        return hash((type(self), self.hilbert, self.dtype))

    def __eq__(self, other):
        if type(self) is type(other):
            return self.hilbert == other.hilbert and self.dtype == other.dtype
        return False

    @property
    def max_conn_size(self) -> int:
        return 1

    @property
    def dtype(self):
        return self._dtype

    def get_conn_padded(self, x):
        x = jnp.asarray(x)

        if isinstance(self.hilbert, SpinOrbitalFermions):
            L = self.hilbert.n_orbitals
            perm = np.concatenate([np.arange(L, 2 * L), np.arange(L)])
            x_conn = x.at[..., None, perm].get(
                unique_indices=True, mode="promise_in_bounds"
            )
            # Fermionic sign: (-1)^(N_up * N_down) = (-1)^N with N_up = N_down = N.
            N = self.hilbert.n_fermions_per_spin[0]
            mel = int((-1) ** N)
            mels = (
                x.at[..., :1].get(unique_indices=True, mode="promise_in_bounds") * 0
                + mel
            ).astype(self.dtype)
        else:
            x_conn = -x[..., None, :]
            # Construct the constant matrix element without materialising new sharded arrays.
            mels = x.at[..., :1].get(unique_indices=True, mode="promise_in_bounds")
            mels = 1 + (mels * 0).astype(self.dtype)

        return x_conn, mels

    def trace(self) -> int:
        if isinstance(self.hilbert, SpinOrbitalFermions):
            # Fixed points: states where each orbital has equal up and down occupation,
            # i.e., x[i] == x[i + L] for all i in 0..L-1. With n_fermions_per_spin=(N,N)
            # these are the C(L, N) states where N orbitals are doubly occupied.
            # Each contributes mel = (-1)^N to the trace.
            L = self.hilbert.n_orbitals
            N = self.hilbert.n_fermions_per_spin[0]
            return int((-1) ** N) * comb(L, N)

        return int(np.any(np.asarray(self.hilbert.local_states) == 0))
