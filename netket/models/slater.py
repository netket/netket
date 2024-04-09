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

import flax.linen as nn
import jax.numpy as jnp

from functools import partial

import numpy as np

from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import NNInitFunc, DType
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax


class Slater2nd(nn.Module):
    r"""
    A slater determinant ansatz for second-quantised spinless or spin-full
    fermions.

    The log-determinants of every spin sub-sector are summed together.

    The total number of parameters of this ansatz is :math:`n_{\mathrm{o}} \times n_{\mathrm{f}}`,
    where :math:`n_{\mathrm{o}}` is the number of orbitals and :math:`n_{\mathrm{f}}` is the
    total number of fermions in the system.
    """

    hilbert: SpinOrbitalFermions
    """The Hilbert space upon which this ansatz is defined. Used to determine the number of orbitals
    and spin subspectors."""

    restricted: bool = True
    """Flag to select the restricted- or unrestricted- Hartree Fock orbtals (Defaults to restricted).

    If restricted, only one set of orbitals are parametrised, and they are used for all spin subsectors.
    If unrestricted, a different set of orbitals are parametrised and used for each spin subsector.
    """
    
    generalized: bool = False
    """Flag to select generalize Hartree-Fock (defaults to standard spin-conserving Hartree-Fock)."""

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

        @partial(jnp.vectorize, signature="(n)->()")
        def log_sd(n):
            # Find the positions of the occupied sites
            R = n.nonzero(size=self.hilbert.n_fermions)[0]
            log_det_sum = 0
            i_start = 0
            
            if self.generalized:
                # extract Nf x Nf submatrix
                A_i = self.orbitals[R,:]
                log_det_sum = nkjax.logdet_cmplx(A_i)
            else:
                for i, (n_fermions_i, M_i) in enumerate(
                    zip(self.hilbert.n_fermions_per_spin, self.orbitals)
                ):
                    # convert global orbital positions to spin-sector-local positions
                    R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
                    # extract the corresponding Nf x Nf submatrix
                    A_i = M_i[R_i]

                    log_det_sum = log_det_sum + nkjax.logdet_cmplx(A_i)
                    i_start = n_fermions_i

            return log_det_sum

        return log_sd(n)
    
class MultiSlater2nd(nn.Module):
    r"""
    A slater determinant ansatz for second-quantised spinless or spin-full
    fermions with a sum of determinants.
    
    Conventions are the same as in `Slater2nd'
    """

    hilbert: SpinOrbitalFermions
    """The Hilbert space upon which this ansatz is defined. Used to determine the number of orbitals
    and spin subspectors."""
    
    n_determinants: int = 1
    """The number of determinants to be summed."""

    restricted: bool = True
    """Flag to select the restricted- or unrestricted- Hartree Fock orbtals (Defaults to restricted).

    If restricted, only one set of orbitals are parametrised, and they are used for all spin subsectors.
    If unrestricted, a different set of orbitals are parametrised and used for each spin subsector.
    """
    
    generalized: bool = False
    """Flag to select generalize Hartree-Fock (defaults to standard spin-conserving Hartree-Fock)."""

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
                f"Number of determinants must be an integer greater than 0."
            )
        # make extra axis with copies to run determinants in parallel
        n_bc = jnp.broadcast_to(n, (self.n_determinants, *n.shape))
        multi_log_det = nn.vmap(
            Slater2nd,
            in_axes=0, out_axes=0, # vmap over copied axis
            variable_axes={'params': 0},
            split_rngs={'params': True}
        )(
            self.hilbert, 
            restricted=self.restricted, 
            generalized=self.generalized, 
            kernel_init=self.kernel_init, 
            param_dtype=self.param_dtype
        )(n_bc)
        # sum the determinants
        log_det_sum = nkjax.logsumexp_cplx(multi_log_det, axis=0)
        return log_det_sum
        
        