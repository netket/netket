# Copyright 2023 The NetKet Authors - All rights reserved.
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

from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import flax.linen as nn

import numpy as np

from netket.hilbert import SpinOrbitalFermions
from netket.utils.types import NNInitFunc, DType
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax


class Slater2nd(nn.Module):
    r"""
    A slater determinant ansatz for second-quantised spinless or spin-full
    fermions.

    When working with spin-full fermionic hilbert spaces (where the number of degrees of
    freedom is a multiple of the number of orbitals) :class:`~netket.models.Slater2nd`
    may behave in 3 different ways, depending on some flags.  Those modes differ by how
    the orbitals are represented.

    The more restrictions we impose, the lower the number of parameters,
    the higher the number of imposed symmetries, but the expressivity will be worse and it will
    be less likely to attain accurate ground-state energies.

    Those options are summarised in this table, while the details are discussed below.

    =================  =====================================================================  =============================================
    Hartree Fock type  Number of Parameters                                                   Options
    =================  =====================================================================  =============================================
    Generalized        :math:`n_{\mathrm{M}} \times n_{\mathrm{f}}`                           :code:`generalized = True`
    Unrestricted       :math:`n_{\mathrm{S}} \times n_{\mathrm{L}} \times n_{\textrm{f, s}}`  :code:`generalized=False, restricted=False`
    Restricted         :math:`n_{\mathrm{L}} \times n_{\textrm{f, s}}`                        :code:`generalized=False, restricted=True`
    =================  =====================================================================  =============================================


    Where we use :math:`n_{\textrm{M}}` to denote the number of fermionic modes, :math:`n_{\textrm{L}}`
    the number of spatial orbitals, and :math:`n_{\textrm{S}}` the number of spin states. The number
    of fermions is denoted :math:`n_{\mathrm{f}}`, or :math:`n_{\textrm{f}, \alpha}` for the number of
    fermions in a given spin sector :math:`\alpha`. We assume the same number of fermions in each spin
    sector for simplicity.


    Details about different hartree Fock types
    ==========================================

    Assume we introduce a set of orbitals :math:`\phi_\mu(r, s)` with orbital index :math:`\mu`.

    - **Generalized Hartree-Fock** (:code:`generalized = True`) is the most general case
      where we impose no restrictions. In particular, we do not restrict the orbitals
      to have definite spin or orbital quantum numbers.
      The total number of parameters is :math:`n_{\mathrm{M}} \times n_{\mathrm{f}}`. Hence,
      any fermion can occupy any of the fermionic modes.
    - **Hartree-Fock (Spin-Conserving)** (:code:`[generalized=False,] restricted=True/False`).
      Most physical Hamiltonians are spin conserving, and hence we can impose it also on the
      wave-function. In this case, we separate the orbital index :math:`\mu \to (l, \alpha)` into
      a spin and spatial orbital part: :math:`\phi_\mu(r, s)=\varphi_{l,\alpha}(r) \chi_{\alpha}(s)`.
      Here, :math:`l` and :math:`\alpha` indicate the orbital and spin quantum numbers associated
      with the orbital, and :math:`(r, s)` are the position vector and spin quantum number at which
      we aim to evaluate the orbital (i.e. properties of a given fermion). Furthermore,
      :math:`\varphi_{l,\alpha}(r)` is the spatial orbital at position :math:`r`, and and
      :math:`\chi_\alpha(s)` the spin part.

      - **Unrestricted Hartree Fock (UHF)** (:code:`[generalized=False,] restricted=False`),
        the orbitals can have a different spatial
        orbital :math:`\varphi` for different spin states. Since e.g. the up
        spin fermions cannot occupy the down spin orbitals and vice versa, the Slater matrix becomes block
        diagonal. This allows us to write the determinant as a product of determinants of the two spin sectors.
        The total number of parameters is :math:`n_{\mathrm{S}} \times n_{\mathrm{L}} \times n_{\textrm{f, s}}`. For
        more information, see
        `Wikipedia: Unrestricted Hartree-Fock <https://en.wikipedia.org/wiki/Unrestricted_Hartree%E2%80%93Fock>`_
      - **Restricted Hartree-Fock (RHF)** (:code:`[generalized=False, restricted=True]`), which assumes
        that different spin states have the same spatial orbitals
        in :math:`\phi_\mu(r, s)=\varphi_l(r) \chi_\alpha(s)`, and hence :math:`\varphi_l` only depends
        on the spatial orbital index :math:`l`. The number of
        parameters now reduces to :math:`n_{\mathrm{L}} \times n_{\textrm{f, s}}`.

    """

    hilbert: SpinOrbitalFermions
    """The Hilbert space upon which this ansatz is defined. Used to determine the number of orbitals
    and spin subspectors."""

    generalized: bool = False
    """Uses Generalized Hartree-Fock if True (defaults to `False`, corresponding to the
    standard spin-conserving Hartree-Fock).
    """

    restricted: bool = True
    """Flag to select the restricted- or unrestricted- Hartree Fock orbitals
    (Defaults to restricted).

    This flag is ignored if :code:`generalized=True`.

    - If restricted, only one set of orbitals are parametrised, and they are
      used for all spin subsectors. This only works if every spin subsector
      holds the same number of fermions.
    - If unrestricted, a different set of orbitals are parametrised and used
      for each spin subsector.
    """

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the orbital parameters."""

    param_dtype: DType = float
    """Dtype of the orbital amplitudes."""

    def __post_init__(self):
        if not isinstance(self.hilbert, SpinOrbitalFermions):
            raise TypeError(
                "Slater2nd only supports 2nd quantised fermionic hilbert spaces."
            )
        if self.hilbert.n_fermions is None:
            raise TypeError(
                "Slater2nd only supports hilbert spaces with a "
                "fixed number of fermions."
            )
        if self.restricted:
            if not all(
                np.equal(
                    self.hilbert.n_fermions_per_spin,
                    self.hilbert.n_fermions_per_spin[0],
                )
            ):
                raise ValueError(
                    "Restricted Hartree Fock only makes sense for spaces with "
                    "same number of fermions on every subspace."
                )
        super().__post_init__()

    def setup(self):
        # Every determinant is a matrix of shape (n_orbitals, n_fermions_i) where
        # n_fermions_i is the number of fermions in the i-th spin sector.
        if self.generalized:
            M = self.param(
                "M",
                self.kernel_init,
                (self.hilbert.size, self.hilbert.n_fermions),
                self.param_dtype,
            )
            self.orbitals = M
        else:
            if self.restricted:
                M = self.param(
                    "M",
                    self.kernel_init,
                    (self.hilbert.n_orbitals, self.hilbert.n_fermions_per_spin[0]),
                    self.param_dtype,
                )

                self.orbitals = [M for _ in self.hilbert.n_fermions_per_spin]
            else:
                self.orbitals = [
                    self.param(
                        f"M_{i}",
                        self.kernel_init,
                        (self.hilbert.n_orbitals, nf_i),
                        self.param_dtype,
                    )
                    for i, nf_i in enumerate(self.hilbert.n_fermions_per_spin)
                ]

    def __call__(self, n):
        """
        Assumes inputs are strings of 0,1 that specify which orbitals are occupied.
        Spin sectors are assumed to follow the SpinOrbitalFermion's factorisation,
        meaning that the first `n_orbitals` entries correspond to sector -1, the
        second `n_orbitals` correspond to 0 ... etc.
        """
        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape} ({n.shape[-1]} dof)."
            )
        if not jnp.issubdtype(n, int):
            n = jnp.isclose(n, 1)

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            # Find the positions of the occupied sites
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            log_det_sum = 0
            i_start = 0

            if self.generalized:
                # extract Nf x Nf submatrix
                A_i = self.orbitals[R, :]
                log_det_sum = nkjax.logdet_cmplx(A_i)
            else:
                for i, (n_fermions_i, M_i) in enumerate(
                    zip(self.hilbert.n_fermions_per_spin, self.orbitals)
                ):
                    if n_fermions_i == 0:
                        continue
                    # convert global orbital positions to spin-sector-local positions
                    R_i = (
                        R[i_start : i_start + n_fermions_i]
                        - i * self.hilbert.n_orbitals
                    )
                    # extract the corresponding Nf x Nf submatrix
                    A_i = M_i[R_i]

                    log_det_sum = log_det_sum + nkjax.logdet_cmplx(A_i)
                    i_start += n_fermions_i

            return log_det_sum

        return log_sd(n)

    def rdm_one_body(self, return_intermediates: bool = False):
        r"""
        Compute the one-body reduced density matrix from the current model parameters.

        For a Slater determinant with orbitals :math:`\phi_\mu`, the density matrix is:

        .. math::

            \rho_{ij} = \sum_{\mu \in \text{occupied}} \phi^*_{i,\mu} \phi_{j,\mu}

        This can be written as :math:`\rho = M (M^\dagger M)^{-1} M^\dagger` where
        :math:`M` is the orbital matrix.

        Args:
            return_intermediates: If True, also return intermediate values needed for
                QGT computation. (Internal)

        Returns:
            The one-body density matrix of shape (n_modes, n_modes)
        """
        if self.generalized:
            M = self.orbitals
            S_inv = jnp.linalg.inv(M.conj().T @ M)
            rdm = M @ S_inv @ M.conj().T
            intermediates = {"S_inv": S_inv, "rdm": rdm}

        elif self.restricted:
            # Restricted (spin-conserving) Hartree-Fock: Same orbitals for all sectors
            M = self.orbitals[0]  # shape (n_orbitals, n_fermions_per_spin)
            S_inv = jnp.linalg.inv(M.conj().T @ M)
            rdm_sector = M @ S_inv @ M.conj().T

            n_spin_sectors = self.hilbert.n_spin_subsectors
            rdm = jsp.linalg.block_diag(*[rdm_sector] * n_spin_sectors)
            intermediates = {"S_inv": S_inv, "rdm_sector": rdm_sector}

        else:
            # Unrestricted (spin-conserving) Hartree-Fock : Different orbitals per sector
            n_spin_sectors = self.hilbert.n_spin_subsectors
            S_invs = []
            rdm_sectors = []

            for i in range(n_spin_sectors):
                M_i = self.orbitals[i]  # shape (n_orbitals, n_fermions_i)
                S_inv_i = jnp.linalg.inv(M_i.conj().T @ M_i)
                rdm_i = M_i @ S_inv_i @ M_i.conj().T
                S_invs.append(S_inv_i)
                rdm_sectors.append(rdm_i)

            rdm = jsp.linalg.block_diag(*rdm_sectors)
            intermediates = {"S_invs": S_invs, "rdm_sectors": rdm_sectors}

        if return_intermediates:
            return rdm, intermediates
        return rdm

    def qgt_by_wick(self) -> jax.Array:
        r"""
        Compute the centered quantum geometric tensor (QGT) for the Slater determinant.

        For a Slater determinant parameterized by orbital coefficients :math:`C_{\mu i}`,
        the centered QGT for general non-orthonormal orbitals is:

        .. math::

            Q_{(\mu i),(\nu j)} = [\delta_{\mu\nu} - P_{\mu\nu}] (S^{-1})_{ji}

        where:
            - :math:`S = C^\dagger C` is the orbital overlap matrix
            - :math:`P = C S^{-1} C^\dagger` is the projector onto occupied orbitals (the 1-RDM)

        In Kronecker notation: :math:`Q = (I - P) \otimes S^{-T}`.

        Equivalently, for orbital variations :math:`dC_1` and :math:`dC_2`:

        .. math::

            \langle d\Psi_1 | d\Psi_2 \rangle_c = \mathrm{Tr}[S^{-1} dC_1^\dagger (I - P) dC_2]

        The QGT is returned as a 2D matrix of shape (n_parameters, n_parameters) where
        n_parameters is the total number of orbital coefficients, matching the standard
        NetKet QGT interface.

        Returns:
            The centered QGT as a 2D matrix of shape (n_parameters, n_parameters).
        """
        _, data = self.rdm_one_body(return_intermediates=True)

        if self.generalized:

            I_minus_rdm = jnp.eye(self.hilbert.size) - data["rdm"]
            Q = jnp.kron(I_minus_rdm, data["S_inv"].T)
            return Q

        elif self.restricted:
            I_minus_rdm_sector = jnp.eye(self.hilbert.n_orbitals) - data["rdm_sector"]
            Q_single_sector = jnp.kron(I_minus_rdm_sector, data["S_inv"].T)
            # Multiply by number of spin sectors since each parameter affects all sectors
            n_spin_sectors = self.hilbert.n_spin_subsectors
            Q = n_spin_sectors * Q_single_sector
            return Q

        else:
            # Unrestricted HF: build block-diagonal QGT
            Q_blocks = []

            for S_inv_i, rdm_i in zip(data["S_invs"], data["rdm_sectors"]):
                I_minus_rdm_i = jnp.eye(self.hilbert.n_orbitals) - rdm_i
                Q_i = jnp.kron(I_minus_rdm_i, S_inv_i.T)
                Q_blocks.append(Q_i)

            return jsp.linalg.block_diag(*Q_blocks)


class MultiSlater2nd(nn.Module):
    r"""
    A slater determinant ansatz for second-quantised spinless or spin-full
    fermions with a sum of determinants.

    Refer to :class:`~netket.experimental.models.Slater2nd` for details about the different
    variants of Hartree Fock and the flags.
    """

    hilbert: SpinOrbitalFermions
    """The Hilbert space upon which this ansatz is defined. Used to determine the number of orbitals
    and spin subspectors."""

    n_determinants: int = 1
    """The number of determinants to be summed."""

    generalized: bool = False
    """Uses Generalized Hartree-Fock if True (defaults to `False`, corresponding to the
    standard spin-conserving Hartree-Fock).
    """

    restricted: bool = True
    """Flag to select the restricted- or unrestricted- Hartree Fock orbitals
    (Defaults to restricted).

    This flag is ignored if :code:`generalized=True`.

    - If restricted, only one set of orbitals are parametrised, and they are
      used for all spin subsectors. This only works if every spin subsector
      holds the same number of fermions.
    - If unrestricted, a different set of orbitals are parametrised and used
      for each spin subsector.
    """

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the orbital parameters."""

    param_dtype: DType = float
    """Dtype of the orbital amplitudes."""

    @nn.compact
    def __call__(self, n):
        """
        Assumes inputs are strings of 0,1 that specify which orbitals are occupied.
        Spin sectors are assumed to follow the SpinOrbitalFermion's factorisation,
        meaning that the first `n_orbitals` entries correspond to sector -1, the
        second `n_orbitals` correspond to 0 ... etc.
        """
        if not self.n_determinants:
            raise ValueError(
                "Number of determinants must be an integer greater than 0."
            )
        # make extra axis with copies to run determinants in parallel
        n_bc = jnp.broadcast_to(n, (self.n_determinants, *n.shape))
        multi_log_det = nn.vmap(
            Slater2nd,
            in_axes=0,
            out_axes=0,  # vmap over copied axis
            variable_axes={"params": 0},
            split_rngs={"params": True},
        )(
            self.hilbert,
            restricted=self.restricted,
            generalized=self.generalized,
            kernel_init=self.kernel_init,
            param_dtype=self.param_dtype,
        )(
            n_bc
        )
        # sum the determinants
        log_det_sum = nkjax.logsumexp_cplx(multi_log_det, axis=0)
        return log_det_sum
