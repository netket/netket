from .metropolis_hastings import *
from ._kernels import _HamiltonianKernel


def MetropolisHamiltonian(machine, hamiltonian, n_chains=16, sweep_size=None, **kwargs):
    r"""
    Sampling based on the off-diagonal elements of a Hamiltonian (or a generic Operator).
    In this case, the transition matrix is taken to be:

    .. math::
       T( \mathbf{s} \rightarrow \mathbf{s}^\prime) = \frac{1}{\mathcal{N}(\mathbf{s})}\theta(|H_{\mathbf{s},\mathbf{s}^\prime}|),

    where :math:`\theta(x)` is the Heaviside step function, and :math:`\mathcal{N}(\mathbf{s})`
    is a state-dependent normalization.
    The effect of this transition probability is then to connect (with uniform probability)
    a given state :math:`\mathbf{s}` to all those states :math:`\mathbf{s}^\prime` for which the Hamiltonian has
    finite matrix elements.
    Notice that this sampler preserves by construction all the symmetries
    of the Hamiltonian. This is in generally not true for the local samplers instead.

    Args:
       machine: A machine :math:`\Psi(s)` used for the sampling.
                The probability distribution being sampled
                from is :math:`F(\Psi(s))`, where the function
                :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
       hamiltonian: The operator used to perform off-diagonal transition.
       n_chains: The number of Markov Chain to be run in parallel on a single process.
       sweep_size: The number of exchanges that compose a single sweep.
                   If None, sweep_size is equal to the number of degrees of freedom (n_visible).


    Examples:
       Sampling from a RBM machine in a 1D lattice of spin 1/2

       >>> import netket as nk
       >>>
       >>> g=nk.graph.Hypercube(length=10,n_dim=2,pbc=True)
       >>> hi=nk.hilbert.Spin(s=0.5,graph=g)
       >>>
       >>> # RBM Spin Machine
       >>> ma = nk.machine.RbmSpin(alpha=1, hilbert=hi)
       >>>
       >>> # Transverse-field Ising Hamiltonian
       >>> ha = nk.operator.Ising(hilbert=hi, h=1.0)
       >>>
       >>> # Construct a MetropolisHamiltonian Sampler
       >>> sa = nk.sampler.MetropolisHamiltonian(machine=ma,hamiltonian=ha)
    """
    return MetropolisHastings(
        machine,
        _HamiltonianKernel(machine, hamiltonian),
        n_chains,
        sweep_size,
        **kwargs,
    )


def MetropolisHamiltonianPt(
    machine, hamiltonian, n_replicas=16, sweep_size=None, **kwargs
):
    r"""
    This sampler performs parallel-tempering
    moves in addition to the local moves implemented in `MetropolisLocal`.
    The number of replicas can be :math:`N_{\mathrm{rep}}` chosen by the user.

    Args:
        machine: A machine :math:`\Psi(s)` used for the sampling.
                  The probability distribution being sampled
                  from is :math:`F(\Psi(s))`, where the function
                  :math:`F(X)`, is arbitrary, by default :math:`F(X)=|X|^2`.
        hamiltonian: The operator used to perform off-diagonal transition.
        n_replicas: The number of replicas used for parallel tempering.
        sweep_size: The number of exchanges that compose a single sweep.
                     If None, sweep_size is equal to the number of degrees of freedom (n_visible).
        batch_size: The batch size to be used when calling log_val on the given Machine.
                    If None, batch_size is equal to the number of replicas (n_replicas).
    """
    return MetropolisHastingsPt(
        machine,
        _HamiltonianKernel(machine, hamiltonian),
        n_replicas,
        sweep_size,
        **kwargs,
    )
