import numpy as np
import sparse

from typing import Sequence
from netket.graph import AbstractGraph
from netket.utils.types import DType

import jax.numpy as jnp

from netket.utils import struct
from netket.hilbert import SpinOrbitalFermions

from . import ParticleNumberAndSpinConservingFermioperator2nd


@struct.dataclass
class FermiHubbardJax(ParticleNumberAndSpinConservingFermioperator2nd):
    r"""
    Fermi-Hubbard Hamiltonian based on the generic ParticleNumberAndSpinConservingFermioperator2nd

    .. math::
        \hat H = -t \sum_{<ij>,\sigma} (\hat c_{i\sigma}^\dagger \hat c_{j\sigma} + h.c.) + U \sum_{i}(\hat n_{i\uparrow} \hat n_{i\downarrow})

    This implementation is more efficient than a FermionOperator2nd created using create,destroy,number from nk.operator.fermion
    as shown in the examples, as it considers only matrix elements with the correct number of particles.
    """

    def __pre_init__(
        self,
        hilbert: SpinOrbitalFermions,
        graph: AbstractGraph,
        t: float | Sequence[float] = 1.0,
        U: float | Sequence[float] = 1.0,
        dtype: DType | None = None,
    ):
        r"""
        Constructs a new FermiHubbardJax operator given a hilbert space, a graph
        specifying the connectivity and the interaction strength.

        Args:
           hilbert: Hilbert space the operator acts on.
           graph: Graph
           t: The strength of the hopping term.
           U: The strength of the on-site coulomb interaction term.
           dtype: The dtype of the matrix elements.

        Examples:
           Constructs a FermiHubbardJax operator for a 2D system at half filling.

           >>> import netket as nk
           >>> import netket.experimental
           >>> g = nk.graph.Hypercube(length=4, n_dim=2, pbc=True)
           >>> hi = nk.hilbert.SpinOrbitalFermions(n_orbitals=g.n_nodes, s=1/2, n_fermions_per_spin=(4,4))
           >>> op = netket.experimental.operator.FermiHubbardJax(hi, t=1.0, U=1.0, graph=g)
        """

        if isinstance(t, Sequence):
            assert len(t) == graph.n_edges
        if isinstance(U, Sequence):
            assert len(U) == graph.n_nodes
        assert hilbert.n_spin_subsectors == 2
        assert hilbert.size == 2 * graph.n_nodes

        t = jnp.asarray(t, dtype=dtype)
        U = jnp.asarray(U, dtype=dtype)

        ij = np.array(graph.edges()).T
        t = np.broadcast_to(t, graph.n_edges)
        t_mat = sparse.COO(ij, t, shape=(graph.n_nodes,) * 2)

        U = np.broadcast_to(U, graph.n_nodes)
        iiii = np.array(
            [
                np.array(graph.nodes()),
            ]
            * 4
        )
        U_mat = sparse.COO(iiii, U, shape=(graph.n_nodes,) * 4)

        operators_sector = {}
        operators_sector[2, (0, 1)] = -(t_mat + t_mat.T)
        operators_sector[4, ((1, 0),)] = U_mat

        op = ParticleNumberAndSpinConservingFermioperator2nd._from_sparse_arrays_normal_order(
            hilbert, operators_sector
        )

        return (op._hilbert, op._operator_data), {}

    @property
    def is_hermitian(self):
        return True
