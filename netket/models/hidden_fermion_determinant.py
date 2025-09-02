import jax.numpy as jnp
import jax
import flax.linen as nn 
import netket as nk
from functools import partial
from typing import Tuple, cast, Callable


@jax.jit
def _log_det(A:jax.Array) -> jax.Array:
    sign, logabsdet = jnp.linalg.slogdet(A)
    return logabsdet.astype(complex) + jnp.log(sign.astype(complex))


@partial(jax.jit, static_argnames = ("n_fermions", "n_hidden_fermions", "n_fermions_per_spin"))
@partial(jax.vmap, in_axes = (0,0, None, None, None, None), out_axes = 0)
def log_vals(n: jax.Array, phi_h: jax.Array, phi_v: Tuple, n_fermions: int, n_hidden_fermions: int,
             n_fermions_per_spin: Tuple[int, ...]) -> jax.Array:
    
        R = n.nonzero(size=n_fermions)[0] # this part gathers the indices of the occupied orbitals for all MC chains
        log_det_sum = 0.0
        i_start = 0
        m_start = 0
        for i, (nf_i, visible_orbitals) in enumerate(
            zip(n_fermions_per_spin, phi_v)
        ):
            # convert global orbital positions to spin-sector-local positions (Nf, )
            R_i = R[i_start : i_start + nf_i] - i * n_fermions # indices of occupied orbitals in different spin sectors

            # extract the corresponding Nf x (Nf + Nh) submatrix
            visible_sub = visible_orbitals[R_i]

            hidden_sub = phi_h[
                m_start : m_start
                + n_hidden_fermions * (nf_i + n_hidden_fermions)
            ]
            hidden_sub = hidden_sub.reshape(
                n_hidden_fermions, -1
            )  # Nh x (Nf + Nh)

            Phi = jnp.concatenate(
                (visible_sub, hidden_sub), axis=0
            )  # (Nf + Nh) x (Nf + Nh)


            log_det_sum = log_det_sum + _log_det(Phi)
            i_start = i_start + nf_i
            m_start = (
                m_start
                + (nf_i + n_hidden_fermions) * n_hidden_fermions
            )

        return jnp.array(log_det_sum)



class HiddenFermionDeterminant(nn.Module):

    """Implementation of the Hidden Fermion Determinant State (HFDS) ansätze, parametrized by a 
    multilayer perceptron (MLP), as described in `https://doi.org/10.1073/pnas.2122059119`.
    
    
    Details 
    =======================================================================

    The HFDS ansätze is defined as follows. Given a system of :math:`N` fermions and :math:`M` modes, we introduce
    :math:`\\tilde{N}` hidden fermions that can occupy :math:`\\tilde{M}` hidden modes. 
    The many-body wavefunction is then defined as a Slater determinant on this augmented space.

    The Slater matrix is then an :math:`(M+ \\tilde{M}) \times (N+ \\tilde{N})` matrix, where the first :math:`M` rows correspond to the physical modes
    and the last :math:`\\tilde{M}` rows correspond to the hidden modes. The rows of this matrix are then sliced according to the occupations
    of the physical and hidden fermions producing a square matrix of shape :math:`(N + \\tilde{N}) \\times (N + \\tilde{N})`.

    In this implementation, the first :math:`M` rows are parametrized by a set of input-independent orbitals. The sliced hidden submatrix of shape 
    :math:`\\tilde{N} \\times (N + \\tilde{N})` is computed from an MLP that takes as input the occupations of the physical fermions. 
    This method hinges on the assumption that the occupations of the hidden modes are a function of the occupations of the physical modes. 

    More precisely, the wavefunction is defined as: 

    :math:`\\psi(n) = \\prod_\\sigma \\det \\left( \\left[\\Phi^{(0), \\sigma)}\\right]_{n} \\right] + \\tilde \\Phi^\\sigma(n) \\right)`

    where :math:`\\left[\\Phi^{(0), \\sigma)}\right]_{n}` is the sliced submatrix of the visible orbitals whose last :math:`\\tilde N` rows are zero and :math:`\\tilde \\Phi^\\sigma(n)` is the hidden submatrix computed from the MLP, whose first :math:`N` rows are zero. The product runs over the spin sectors of the system, i.e the full Slater matrix is block diagonal in the spin sectors.

    """


    hilbert: nk.hilbert.SpinOrbitalFermions
    """The Hilbert space upon which the ansätze is defined, used to extract the 
    number of fermions and orbitals."""

    n_hidden_fermions: int
    """Number of hidden fermions. If the system has multiple spin sectors, this number is 
    to be understood as the number of hidden fermions in each spin sector."""

    hidden_unit_density: float | int = 1.0
    """Density of hidden units in the feedforward network, this is the number 
    of hidden units per physical orbital."""
    
    kernel_init: Callable = nn.initializers.glorot_normal()
    """Initializer for the weights of the feedforward network and the visible orbitals."""

    param_dtype: type = jnp.float64
    """Datatype of the parameters."""


    def __post_init__(self):
        if not isinstance(self.hilbert, nk.hilbert.SpinOrbitalFermions):
            raise TypeError(
                "Only 2nd quantized fermionic Hilbert spaces are supported."
            )
        if self.hilbert.n_fermions is None:
            raise TypeError(
                "Only Hilbert spaces with a fixed number of fermions are supported."
            )
        
        if any(nf_i is None for nf_i in self.hilbert.n_fermions_per_spin):
            raise ValueError(
                "Only hilbert spaces with a fixed number of fermions per spin sector are supported."
            )
        
        # Cast the tuple to a tuple of integers to satisfy the type checker
        self.n_fermions_per_spin = cast(Tuple[int, ...], self.hilbert.n_fermions_per_spin)
            
        
        super().__post_init__()

    
    def setup(self):

        self.visible_orbitals = [
            self.param(f"phi_v_{i}", self.kernel_init,
                (self.hilbert.n_orbitals, nf_i + self.n_hidden_fermions),
                self.param_dtype,) for i,nf_i in enumerate(self.n_fermions_per_spin)]
        

        self.output_dim = sum([
            self.n_hidden_fermions*(nf_i + self.n_hidden_fermions) for nf_i in self.n_fermions_per_spin]
        ) #shape of the hidden submatrix

        self.ff = nn.Sequential(
            [
                nn.Dense(int(self.hilbert.n_orbitals * self.hidden_unit_density),
                         dtype=self.param_dtype, param_dtype=self.param_dtype,
                         kernel_init=self.kernel_init),
                nn.tanh,
                nn.Dense(self.output_dim,
                         dtype=self.param_dtype, param_dtype=self.param_dtype,
                         kernel_init=self.kernel_init),
            ]
        )


    def __call__(self, n: jax.Array) -> jax.Array:

        """
        Assumes inputs are strings of 0,1 that specify which orbitals are occupied.
        Spin sectors are assumed to follow the SpinOrbitalFermion's factorization,
        meaning that the first `n_orbitals` entries correspond to sector -1, the
        second `n_orbitals` correspond to 0 ... etc.
        """
        

        if not n.shape[-1] == self.hilbert.size:
            raise ValueError(
                f"Dimension mismatch. Expected samples with {self.hilbert.size} "
                f"degrees of freedom, but got a sample of shape {n.shape} ({n.shape[-1]} dof)."
            )
        
        phi_h = self.ff(n) #extract the hidden submatrices for all spin sectors
        psi = log_vals(n, phi_h, self.visible_orbitals, 
                       self.hilbert.n_fermions,
                       self.n_hidden_fermions,
                       self.n_fermions_per_spin)

        return psi

