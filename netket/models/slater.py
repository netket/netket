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

from netket.experimental.hilbert import SpinOrbitalFermions
from netket.utils.types import NNInitFunc, DType
from netket.nn.masked_linear import default_kernel_init
from netket import jax as nkjax


def _log_det(A):
    """Complex-aware logdeterminant built on top of
    {func}`jax.numpy.linalg.slogdet` and combining the sign again.
    """
    sign, logabsdet = jnp.linalg.slogdet(A)
    cplx_type = nkjax.dtype_complex(A.dtype)
    return logabsdet.astype(cplx_type) + jnp.log(sign.astype(cplx_type))


class LogSlater2nd(nn.Module):
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
    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the orbital parameters."""
    param_dtype: DType = float
    """Dtype of the orbital amplitudes."""

    def __post_init__(self):
        if not isinstance(self.hilbert, SpinOrbitalFermions):
            raise TypeError(
                "LogSlater2nd only supports 2nd quantised fermionic hilbert spaces."
            )
        if self.hilbert.n_fermions is None:
            raise TypeError(
                "LogSlater2nd only supports hilbert spaces with a "
                "fixed number of fermions."
            )
        super().__post_init__()

    def setup(self):
        # Every determinant is a matrix of shape (n_orbitals, n_fermions_i) where
        # n_fermions_i is the number of fermions in the i-th spin sector.
        self.determinants = [
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
            for i, (n_fermions_i, M_i) in enumerate(
                zip(self.hilbert.n_fermions_per_spin, self.determinants)
            ):
                # convert global orbital positions to spin-sector-local positions
                R_i = R[i_start : i_start + n_fermions_i] - i * self.hilbert.n_orbitals
                # extract the corresponding Nf x Nf submatrix
                A_i = M_i[R_i]

                log_det_sum = log_det_sum + _log_det(A_i)
                i_start = n_fermions_i

            return log_det_sum

        return log_sd(n)
