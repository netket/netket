from functools import partial
from typing import Union

import numpy as np
import sparse

import jax
import jax.numpy as jnp

from flax import struct

from netket.operator import DiscreteJaxOperator
from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import Array
from netket.operator import FermionOperator2nd, FermionOperator2ndJax
from netket.utils.optional_deps import import_optional_dependency

from .._pyscf_utils import (
    TV_from_pyscf_molecule,
    to_desc_order_sparse,
    compute_pyscf_integrals,
)

from .._normal_order_utils import (
    to_normal_order,
    to_normal_order_sector,
    SpinOperatorArrayDict,
)
from ._conversion import (
    fermiop_to_pnc_format_helper,
    fermiop_to_pnc_format_spin_helper,
    pnc_format_to_fermiop_helper,
)
from ._operator_data import (
    collect_ops,
    sparse_arrays_to_coords_data_dict,
    prepare_operator_data_from_coords_data_dict,
    prepare_operator_data_from_coords_data_dict_spin,
    to_coords_data_sector,
    PNCOperatorDataCollectionDict,
    PNCOperatorArrayDict,
    CoordsDataDictSectorCooArrayType,
    CoordsDataDictSectorType,
)
from ._kernels import get_conn_padded_pnc, get_conn_padded_pnc_spin


@struct.dataclass
class ParticleNumberConservingFermioperator2nd(DiscreteJaxOperator):
    r"""
    Particle-number conserving fermionc operator

    .. math::

        \hat H = w + \sum_{ij} w_{ij} \hat c_i^\dagger \hat c_j + \sum_{ijkl} w_{ijkl} \hat c_i^\dagger \hat c_j^\dagger \hat c_k \hat c_l + \sum_{ijklmn} w_{ijklmn} \hat c_i^\dagger \hat c_j^\dagger c_k^\dagger \hat c_l \hat c_m \hat c_n + \dots

    To be used with netket.hilbert.SpinOrbitalFermions with a fixed number of fermions.

    It uses a custom sparse internal representation,
    please refer to the docstrings of prepare_data and prepare_data_diagonal for details.

    We provide several factory methods to create this operator:
        - ParticleNumberConservingFermioperator2nd.from_fermionoperator2nd:
               Conversion form FermionOperator2nd/FermionOperator2ndJax
        - ParticleNumberConservingFermioperator2nd.from_sparse_arrays:
                From sparse arrays (w, w_ij, w_ijkl, w_ijklmn, ...)
        - ParticleNumberConservingFermioperator2nd.from_pyscf_molecule:
                From pyscf

    Furthermore it can be converted to FermionOperator2nd/FermionOperator2ndJax using the .to_fermionoperator2nd() method.
    """

    # factory methods for internal use only:
    # - ParticleNumberConservingFermioperator2nd._from_coords_data_normal_order:
    #         From tuples of (sites, daggers, weights) representing w, w_ij, ...
    #         where only the lower triangular part is nonzero
    # - ParticleNumberConservingFermioperator2nd._from_sparse_arrays_normal_order:
    #         From sparse arrays (w, w_ij, w_ijkl, w_ijklmn) where i>=j, i>=j>=k>=l etc,
    #         and only the lower triangular part is nonzero

    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: (
        PNCOperatorDataCollectionDict  # custom sparse internal representation
    )

    def get_conn_padded(self, x):
        return get_conn_padded_pnc(self._operator_data, x, self._hilbert.n_fermions)

    @property
    def max_conn_size(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.shape[-1]

    @property
    def dtype(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.dtype

    @property
    def is_hermitian(self):
        # TODO more efficient implementation
        return self.to_fermionoperator2nd().is_hermitian

    @classmethod
    def _from_coords_data_normal_order(
        cls,
        hilbert: SpinOrbitalFermions,
        coords_data_dict: PNCOperatorArrayDict,
        **kwargs,
    ):
        r"""
        initialize from PNCOperatorArrayDict

        used internally.
        """
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        n_orbitals = hilbert.n_orbitals * hilbert.n_spin_subsectors
        data = prepare_operator_data_from_coords_data_dict(
            coords_data_dict, n_orbitals, **kwargs
        )
        return cls(hilbert, data)

    @classmethod
    def _from_sparse_arrays_normal_order(
        cls,
        hilbert: SpinOrbitalFermions,
        operators: list[Union[Array, sparse.COO]],
        **kwargs,
    ):
        r"""
        initialize from a list of arrays in normal order (descending)

        Args:
            hilbert: hilbert space
            Operators: list of dense or sparse arrays, each representing an m-body operator for different m

        Example:
        Given an array A of rank 2m with shape (n_orbitals,)*(2m) this initializes the operator

        .. math::

            \hat A = \sum_{i_1 > \dots > i_m, j_1 > \dots > j_m} A_{i_1 \cdots i_m j_1 \cdots j_m} \hat c_{i_1}^\dagger \cdots \hat c_{i_m}^\dagger \hat c_{j_1} \cdots \hat c_{j_m}

        Throws an error if the arrays are not in descending order.
        """
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
    def from_sparse_arrays(
        cls,
        hilbert: SpinOrbitalFermions,
        operators: list[Union[Array, sparse.COO]],
        **kwargs,
    ):
        r"""
        initialize from a list of arrays

        Args:
            hilbert: hilbert space
            operators: list of dense or sparse arrays, each representing an m-body operator for different m
            cutoff: cutoff to use when converting the operators to the internal format.
                    Use a small but nonzero number, to allow for internal equality checks between arrays.

        Example:
        Given an array `A` of rank 2m with shape `(n_orbitals,)*(2m)` this initializes the operator

        .. math::

            \hat A = \sum_{i_1,\dots,i_m, j_1,\dots,j_m} A_{i_1 \cdots i_m j_1 \cdots j_m} \hat c_{i_1}^\dagger \cdots \hat c_{i_m}^\dagger \hat c_{j_1} \cdots \hat c_{j_m}

        """
        # daggers on the left, but not necessarily desc order

        # check shapes
        n_orbitals = hilbert.size
        for op in operators:
            if hasattr(op, "shape") and op.ndim > 0:  # >= 1-body
                s = op.shape
                ndim = op.ndim
                if ndim % 2 != 0:
                    raise ValueError(
                        "operator array has incompatible number of dimensions"
                    )
                if set(s) != {n_orbitals}:
                    raise ValueError(
                        f"inconsistent operator array shapes, {n_orbitals=} but got shape {s}"
                    )

        ops = collect_ops(operators)
        cutoff = kwargs.get("cutoff", 0)
        ops = jax.tree_util.tree_map(partial(to_desc_order_sparse, cutoff=cutoff), ops)
        terms = sparse_arrays_to_coords_data_dict(ops)
        return cls._from_coords_data_normal_order(hilbert, terms, **kwargs)

    @classmethod
    def from_fermionoperator2nd(
        cls, ha: Union[FermionOperator2nd, FermionOperator2ndJax], **kwargs
    ):
        """
        Convert from FermionOperator2nd

        Args:
            ha : the original FermionOperator2nd/FermionOperator2ndJax operator

        Throws an error if the original operator is not particle-number conserving.
        """
        # ha = ha.to_normal_order()
        t = fermiop_to_pnc_format_helper(ha.terms, ha.weights)
        t = to_normal_order(t)
        terms = {k: (v[0], v[2]) for k, v in t.items()}  # drop daggers
        return cls._from_coords_data_normal_order(ha.hilbert, terms, **kwargs)

    def to_fermionoperator2nd(
        self, _cls=FermionOperator2ndJax
    ) -> FermionOperator2ndJax:
        r"""
        Convert to FermionOperator2ndJax
        """
        terms = []
        weights = []
        for d in self._operator_data.values():
            for k, v in d.items():
                t, w = pnc_format_to_fermiop_helper(*v)
                terms = terms + t.tolist()
                weights = weights + w.tolist()
        return _cls(self._hilbert, terms, weights)

    @classmethod
    def from_pyscf_molecule(
        cls,
        mol: "pyscf.gto.mole.Mole",  # noqa: F821
        mo_coeff: Array,
        cutoff: float = 1e-11,
        **kwargs,
    ):
        r"""
        Constructs the operator from a pyscf molecule

        Args:
            mol: pyscf molecule
            mo_coeff: molecular orbital coefficients, e.g. obtained from a HF calculation
            cutoff: cutoff to use when converting the operators to the internal format.
                    Use a small but nonzero number, to allow for internal equality checks between arrays.
        """
        n_orbitals = int(mol.nao)
        hi = SpinOrbitalFermions(n_orbitals, s=1 / 2, n_fermions_per_spin=mol.nelec)
        E_nuc, Tij, Vijkl = TV_from_pyscf_molecule(mol, mo_coeff, cutoff=cutoff)
        return cls._from_sparse_arrays_normal_order(
            hi, [E_nuc, Tij, 0.5 * Vijkl], **kwargs
        )


@struct.dataclass
class ParticleNumberAndSpinConservingFermioperator2nd(DiscreteJaxOperator):
    r"""
    Particle-number conserving and spin-Z-conserving fermionc operator

    .. math::

        \hat H = w + \sum_{ij \sigma} w_{ij \sigma} \hat c_{i \sigma}^\dagger \hat c_{j \sigma} + \sum_{ijkl\sigma \rho} w_{ijkl\sigma\rho}  \hat c_{i\sigma}^\dagger \hat c_{j \rho}^\dagger \hat c_{k \rho} \hat c_{l \sigma}


    Limited to 2-body operators if acting on > 1 sector at a time

    To be used with netket.hilbert.SpinOrbitalFermions with a fixed number of fermions.

    It uses a custom sparse internal representation,
    please refer to the docstrings of prepare_data and prepare_data_diagonal for details.

    We provide several factory methods to create this operator:
        - ParticleNumberAndSpinConservingFermioperator2nd.from_fermionoperator2nd:
                Conversion form FermionOperator2nd/FermionOperator2ndJax (if possible)
        - ParticleNumberConservingFermioperator2nd.from_pyscf_molecule:
                From pyscf
    Furthermore it can be converted to FermionOperator2nd/FermionOperator2ndJax using the .to_fermiop method.
    """

    # factory methods for internal use only:
    # - ParticleNumberAndSpinConservingFermioperator2nd._from_sites_sectors_daggers_weights:
    #         From a dictionary of tuples {k: (sites, sectors, daggers, weights)} representing w, w_ij\sigma, w_ijkl\sigma
    # - ParticleNumberAndSpinConservingFermioperator2nd._from_sparse_arrays_normal_order_all_sectors:
    #         From sparse arrays for w, w_ij and w_ijkl summing over all possible values of \sigma,\rho
    # - ParticleNumberAndSpinConservingFermioperator2nd._from_coords_data:
    #         From a dictionary of tuples {(k, sectors): (sites, daggers, weights)} representing w, w_ij\sigma, w_ijkl\sigma\rho

    _hilbert: SpinOrbitalFermions = struct.field(pytree_node=False)
    _operator_data: PNCOperatorDataCollectionDict

    @property
    def dtype(self):
        x = jax.ShapeDtypeStruct((1, self._hilbert.size), dtype=jnp.uint8)
        _, mels = jax.eval_shape(self.get_conn_padded, x)
        return mels.dtype

    @property
    def is_hermitian(self):
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
    def _from_coords_data(
        cls, hilbert: SpinOrbitalFermions, coords_data_sectors: CoordsDataDictSectorType
    ):
        assert isinstance(hilbert, SpinOrbitalFermions)
        assert hilbert.n_fermions is not None
        assert hilbert.n_spin_subsectors >= 2
        n_orbitals = hilbert.n_orbitals
        operator_data = prepare_operator_data_from_coords_data_dict_spin(
            coords_data_sectors, n_orbitals
        )
        return cls(hilbert, operator_data)

    @classmethod
    def _from_sparse_arrays_normal_order(
        cls,
        hilbert: SpinOrbitalFermions,
        operators_sector: CoordsDataDictSectorCooArrayType,
    ):
        r"""
        Args:

            operators_sector:
                dict {key: value}
                    key: a tuple (k, sectors)
                         where k is the number of c/c^\dagger
                         and sectors is a tuple of tuples/numbers containing the index / sets
                         of indices of sectors the operator is acting on each element
                         of sectors needs to be ordered in descending order
                    value: a sparse matrix of coefficeints of shape (n_orbitals,)*k
        """
        coords_data_sectors = sparse_arrays_to_coords_data_dict(operators_sector)
        return cls._from_coords_data(hilbert, coords_data_sectors)

    @classmethod
    def _from_sparse_arrays_normal_order_all_sectors(
        cls,
        hilbert: SpinOrbitalFermions,
        operators: list[Union[Array, sparse.COO]],
        cutoff=1e-11,
    ):
        r"""
        Construct the operator from  sparse arrays for :math:`w, w_ij` and :math:`w_ijkl` summing over all possible values of :math:`\sigma,\rho`,

        .. math::
            \hat H = w + \sum_{ij\sigma} w_{ij} c_{i\sigma}^\dagger c_{j\sigma} + \sigma_{ijkl\sigma\rho} w_{ijkl} c_{i\sigma}^\dagger c_{j\rho}^\dagger c_{k\rho} c_{l\sigma}

        Args:
            hilbert: hilbert space
            operators: a list of sparse arrays representing :math:`w, w_ij` and :math:`w_ijkl`
            cutoff: cutoff on the matirx elements
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
                # \sigma_{\sigma!=\rho} c_ijkl  c_i\sigma^\dagger c_j\rho^\dagger c_k\rho c_l\sigma =  \sigma_{\sigma>\rho} (c_ijkl + c_jilk) c_i\sigma^\dagger c_j\rho^\dagger c_k\rho c_l\sigma
                operators_sector[4, sectors2] = v.swapaxes(2, 3) + v.swapaxes(0, 1)
            else:
                raise NotImplementedError
        return cls._from_sparse_arrays_normal_order(hilbert, operators_sector)

    @classmethod
    def from_pyscf_molecule(
        cls,
        mol: "pyscf.gto.mole.Mole",  # noqa: F821
        mo_coeff: Array,
        cutoff: float = 1e-11,
    ):
        r"""
        Constructs the operator from a pyscf molecule

        Args:
            mol: pyscf molecule
            mo_coeff: molecular orbital coefficients, e.g. obtained from a HF calculation
            cutoff: cutoff to use when converting the operators to the internal format.
                    Use a small but nonzero number, to allow for internal equality checks between arrays.
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
    def _from_sites_sectors_daggers_weights(
        cls,
        hilbert: SpinOrbitalFermions,
        t: SpinOperatorArrayDict,
        cutoff: float = 1e-11,
    ):
        r"""
        Initialize from SpinOperatorArrayDict
        Args:
            hilbert : hilbert space
            t: { size : (sites, sectors, daggers, weights) }
            cutoff: cutoff to use when converting the operators to the internal format.
                    Use a small but nonzero number, to allow for internal equality checks between arrays.

        Supports arbitrary order of sites, sectors, and daggers,
        it is internally converted to the right order for the operator
        """
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        tno = to_normal_order_sector(t, n_spin_subsectors, n_orbitals)
        coords_data = to_coords_data_sector(
            tno, n_spin_subsectors, n_orbitals, cutoff=cutoff
        )
        return cls._from_coords_data(hilbert, coords_data)

    @classmethod
    def from_fermionoperator2nd(
        cls, ha: Union[FermionOperator2nd, FermionOperator2ndJax], cutoff: float = 1e-11
    ):
        r"""
        Convert from FermionOperator2nd

        Args:
            ha : the original FermionOperator2nd/FermionOperator2ndJax operator
            cutoff: cutoff to use when converting the operators to the internal format.
                    Use a small but nonzero number, to allow for internal equality checks between arrays.

        Throws an error if the original operator is not particle-number conserving, or spin-Z-conserving.
        """
        hilbert = ha.hilbert
        n_orbitals = hilbert.n_orbitals
        n_spin_subsectors = hilbert.n_spin_subsectors
        t = fermiop_to_pnc_format_spin_helper(
            ha.terms, ha.weights, n_orbitals, n_spin_subsectors
        )
        return cls._from_sites_sectors_daggers_weights(hilbert, t, cutoff=cutoff)
