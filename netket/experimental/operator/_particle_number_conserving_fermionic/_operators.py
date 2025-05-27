from functools import partial

import numpy as np

import jax
import jax.numpy as jnp

from flax import struct

from netket.operator import DiscreteJaxOperator
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import PyTree
from netket.operator import FermionOperator2ndJax
from netket.utils.optional_deps import import_optional_dependency

from .._pyscf_utils import (
    TV_from_pyscf_molecule,
    to_desc_order_sparse,
    compute_pyscf_integrals,
)

from .._normal_order_utils import to_normal_order, to_normal_order_sector
from ._conversion import (
    fermiop_to_pnc_format,
    pnc_to_fermiop_format,
    to_fermiop_helper,
)
from ._operator_data import (
    collect_ops,
    sparse_arrays_to_coords_data_dict,
    prepare_operator_data_from_coords_data_dict,
    prepare_operator_data_from_coords_data_dict_spin,
    to_coords_data_sector,
)
from ._kernels import get_conn_padded_pnc, get_conn_padded_pnc_spin


@struct.dataclass
class ParticleNumberConservingFermioperator2ndJax(DiscreteJaxOperator):
    """
    Particle-number conserving fermionc operator
    H = w + Σ_ij w_ij c_i^† c_j + Σ_ijkl w_ijkl  c_i^† c_j^† c_k c_l + Σ_ijklmn w_ijklmn c_i^† c_j^† c_k^† c_l c_m c_n + ...

    Version without spin.

    To be used with netket.hilbert.SpinOrbitalFermions with a fixed number of fermions.

    It uses a custom sparse internal representation,
    please refer to the docstrings of prepare_data and prepare_data_diagonal for details.

    We provide several factory methods to create this operator:
        - ParticleNumberConservingFermioperator2ndJax.from_fermiop:
               Conversion form FermionOperator2nd/FermionOperator2ndJax
        - ParticleNumberConservingFermioperator2ndJax.from_sparse_arrays:
                From sparse arrays (w, w_ij, w_ijkl, w_ijklmn, ...)
        - ParticleNumberConservingFermioperator2ndJax.from_pyscf_molecule:
                From pyscf

    Furthermore it can be converted to FermionOperator2nd/FermionOperator2ndJax using the .to_fermiop method.
    """

    # factory methods for internal use only:
    # - ParticleNumberConservingFermioperator2ndJax._from_coords_data_normal_order:
    #         From tuples of (sites, daggers, weights) representing w, w_ij, ...
    #         where only the lower triangular part is nonzero
    # - ParticleNumberConservingFermioperator2ndJax._from_sparse_arrays_normal_order:
    #         From sparse arrays (w, w_ij, w_ijkl, w_ijklmn) where i>=j, i>=j>=k>=l etc,
    #         and only the lower triangular part is nonzero

    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree  # custom sparse internal representation

    def get_conn_padded(self, x):
        return get_conn_padded_pnc(self._operator_data, x, self._hilbert.n_fermions)

    @property
    def max_conn_size(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.shape[-1]

    @property
    def dtype(self):
        return NotImplemented
        # return list(self._operator_data.values())[0][2].dtype

    @property
    def is_hermitian(self):
        # TODO more efficient implementation
        return self.to_fermiop().is_hermitian

    @classmethod
    def _from_coords_data_normal_order(cls, hilbert, coords_data_dict, **kwargs):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        n_orbitals = hilbert.n_orbitals * hilbert.n_spin_subsectors
        data = prepare_operator_data_from_coords_data_dict(
            coords_data_dict, n_orbitals, **kwargs
        )
        return cls(hilbert, data)

    @classmethod
    def _from_sparse_arrays_normal_order(cls, hilbert, operators, **kwargs):
        terms = sparse_arrays_to_coords_data_dict(collect_ops(operators))

        for k, v in terms.items():
            if k <= 2:
                pass
            idx = v[0]
            idx_create = idx[:, : idx.shape[1] // 2]
            idx_destroy = idx[:, idx.shape[1] // 2 :]
            for idx_arr in idx_destroy, idx_create:
                if (jnp.diff(idx_arr) > 0).any():
                    raise ValueError("Input arrays are not in normal order")

        return cls._from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_sparse_arrays(cls, hilbert, operators, **kwargs):
        # daggers on the left, but not necessarily desc order
        ops = collect_ops(operators)
        cutoff = kwargs.get("cutoff", 0)
        ops = jax.tree_util.tree_map(partial(to_desc_order_sparse, cutoff=cutoff), ops)
        terms = sparse_arrays_to_coords_data_dict(ops)
        return cls._from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_fermiop(cls, ha, **kwargs):
        # ha = ha.to_normal_order()
        t = fermiop_to_pnc_format(ha.terms, ha.weights)
        t = to_normal_order(t)
        terms = {k: (v[0], v[2]) for k, v in t.items()}  # drop daggers
        return cls._from_coords_data_normal_order(ha.hilbert, terms, **kwargs)

    def to_fermiop(self, cls=FermionOperator2ndJax):
        terms = []
        weights = []
        for d in self._operator_data:
            for k, v in d.items():
                t, w = to_fermiop_helper(*v)
                terms = terms + t.tolist()
                weights = weights + w.tolist()
        return cls(self._hilbert, terms, weights)

    @classmethod
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11, **kwargs):
        n_orbitals = int(mol.nao)
        hi = SpinOrbitalFermions(n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec)
        E_nuc, Tij, Vijkl = TV_from_pyscf_molecule(mol, mo_coeff, cutoff=cutoff)
        return cls._from_sparse_arrays_normal_order(
            hi, [E_nuc, Tij, 0.5 * Vijkl], **kwargs
        )


# TODO generalize it to >4 fermionic operators
@struct.dataclass
class ParticleNumberConservingFermioperator2ndSpinJax(DiscreteJaxOperator):
    """
    Particle-number conserving fermionc operator
    H = w + Σ_ijσ w_ijσ c_iσ^† c_jσ + Σ_ijklσρ w_ijklσρ  c_iσ^† c_jρ^† c_kρ c_lσ

    Version with spin.

    To be used with netket.hilbert.SpinOrbitalFermions with a fixed number of fermions.

    It uses a custom sparse internal representation,
    please refer to the docstrings of prepare_data and prepare_data_diagonal for details.

    We provide several factory methods to create this operator:
        - ParticleNumberConservingFermioperator2ndSpinJax.from_fermiop:
                Conversion form FermionOperator2nd/FermionOperator2ndJax (if possible)
        - ParticleNumberConservingFermioperator2ndSpinJax.from_sparse_arrays:
                From sparse arrays for w, w_ij and w_ijkl specifying the spin sectors σ,ρ explicitly
        - ParticleNumberConservingFermioperator2ndJax.from_pyscf_molecule:
                From pyscf
    Furthermore it can be converted to FermionOperator2nd/FermionOperator2ndJax using the .to_fermiop method.
    """

    # factory methods for internal use only:
    # - ParticleNumberConservingFermioperator2ndSpinJax._from_sites_sectors_daggers_weights:
    #         From a dictionary of tuples {k: (sites, sectors, daggers, weights)} representing w, w_ijσ, w_ijklσ
    # - ParticleNumberConservingFermioperator2ndSpinJax._from_sparse_arrays_normal_order_all_sectors:
    #         From sparse arrays for w, w_ij and w_ijkl summing over all possible values of σ,ρ
    # - ParticleNumberConservingFermioperator2ndSpinJax._from_coords_data:
    #         From a dictionary of tuples {(k, sectors): (sites, daggers, weights)} representing w, w_ijσ, w_ijklσρ

    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PyTree

    @property
    def dtype(self):
        return NotImplemented

    @property
    def is_hermitian(self):
        # TODO actually check it is
        # return True
        return NotImplemented

    @property
    def max_conn_size(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.shape[-1]

    def get_conn_padded(self, x):
        return get_conn_padded_pnc_spin(
            self._operator_data, x, self._hilbert.n_fermions_per_spin
        )

    @classmethod
    def _from_coords_data(cls, hilbert, coords_data_sectors):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        assert hilbert.n_spin_subsectors >= 2
        n_orbitals = hilbert.n_orbitals
        operator_data = prepare_operator_data_from_coords_data_dict_spin(
            coords_data_sectors, n_orbitals
        )
        return cls(hilbert, operator_data)

    @classmethod
    def _from_sparse_arrays(cls, hilbert, operators_sector):
        """ """
        # TODO come up with some interface to specify the sectors
        # then convert it to normal order here
        # alternatively expose _from_sites_sectors_daggers_weights

        raise NotImplementedError

    @classmethod
    def _from_sparse_arrays_normal_order(cls, hilbert, operators_sector):
        """
        Args:
            operators_sector:
                dict {key: value}
                    key: a tuple (k, sectors)
                         where k is the number of c/c^†
                         and sectors is a tuple of tuples/numbers containing the index / sets
                         of indices of sectors the operator is acting on each element
                         of sectors needs to be ordered in descending order
                    value: a sparse matrix of coefficeints of shape (n_orbitals,)*k
        """

        coords_data_sectors = sparse_arrays_to_coords_data_dict(operators_sector)
        return cls._from_coords_data(hilbert, coords_data_sectors)

    @classmethod
    def _from_sparse_arrays_normal_order_all_sectors(
        cls, hilbert, operators, cutoff=1e-11
    ):
        """
        Construct the operator from  sparse arrays for w, w_ij and w_ijkl summing over all possible values of σ,ρ,

        H = w + Σ_ijσ w_ij c_iσ^† c_jσ + Σ_ijklσρ w_ijkl c_iσ^† c_jρ^† c_kρ c_lσ

        Args:
            hilbert: hilbert space
            operators: a list of sparse arrays representing w, w_ij and w_ijkl
            cutoff: cutoff on the matirx elements, default=1e-11
        """
        # operators = [const, hij, hijkl]
        # ops = {0: const, 2: hij_sparse, 4: hijkl_sparse}
        ops = collect_ops(operators)

        operators_sector = {}
        sectors0 = ()
        sectors1 = tuple(np.arange(hilbert.n_spin_subsectors).tolist())
        sectors2 = tuple(
            map(
                tuple,
                np.array(np.tril_indices(hilbert.n_spin_subsectors, -1)).T.tolist(),
            )
        )
        for k, v in ops.items():
            if k == 0:
                operators_sector[0, sectors0] = v
            elif k == 2:
                operators_sector[2, sectors1] = v
            elif k == 4:
                operators_sector[4, sectors1] = to_desc_order_sparse(v, cutoff)
                # add c_ijkl + c_jilk
                # Σ_{σ!=ρ} c_ijkl  c_iσ^† c_jρ^† c_kρ c_lσ =  Σ_{σ>ρ} (c_ijkl + c_jilk) c_iσ^† c_jρ^† c_kρ c_lσ
                operators_sector[4, sectors2] = v.swapaxes(2, 3) + v.swapaxes(0, 1)
            else:
                raise NotImplementedError
        return cls._from_sparse_arrays_normal_order(hilbert, operators_sector)

    @classmethod
    def from_pyscf_molecule(cls, mol, mo_coeff, cutoff=1e-11):
        """
        Constructs the operator from a pyscf molecule

        Args:
            mol: pyscf molecule
            mo_coeff: coefficients
                e.g. run
        """

        sparse = import_optional_dependency("sparse")

        n_orbitals = int(mol.nao)
        hilbert = SpinOrbitalFermions(
            n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec
        )

        const, hij, hijkl = compute_pyscf_integrals(
            mol, mo_coeff
        )  # not in normal order
        hij = hij * (jnp.abs(hij) > cutoff)
        hij_sparse = sparse.COO.from_numpy(hij)
        hijkl = hijkl * (jnp.abs(hijkl) > cutoff)
        hijkl_sparse = 0.5 * sparse.COO.from_numpy(hijkl)
        return cls._from_sparse_arrays_normal_order_all_sectors(
            hilbert, [const, hij_sparse, hijkl_sparse], cutoff=cutoff
        )

    @classmethod
    def _from_sites_sectors_daggers_weights(cls, hilbert, t, cutoff=1e-11):
        # t: { size : (sites, sectors, daggers, weights) }
        # arbitrary order of sites, sectors, and daggers
        # is internally converted to the right order for the operator
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        tno = to_normal_order_sector(t, n_spin_subsectors, n_orbitals)
        coords_data = to_coords_data_sector(
            tno, n_spin_subsectors, n_orbitals, cutoff=cutoff
        )
        return cls._from_coords_data(hilbert, coords_data)

    @classmethod
    def from_fermiop(cls, ha, cutoff=1e-11):
        hilbert = ha.hilbert
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        t = pnc_to_fermiop_format(
            ha.terms, ha.weights, n_orbitals, n_spin_subsectors
        )
        return cls._from_sites_sectors_daggers_weights(hilbert, t, cutoff=cutoff)
