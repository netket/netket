import os
os.environ['NETKET_EXPERIMENTAL_SHARDING'] = '1'

import jax
jax.distributed.initialize()

import matplotlib.pyplot as plt

import netket as nk

import jax.numpy as jnp
from netket.models.transformer import ViT

seed = 0
key = jax.random.key(seed)

L = 10
n_dim = 2
J2 = 0.5

lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = 0.25 * nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

# Initialize the ViT variational wave function
vit_module = ViT(
    num_layers=4, 
    d_model=72, 
    n_heads=12, 
    patch_size=2, 
    transl_invariant=True,
    spatial_attention=True #* the default implementation works on square lattice with PBC
)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, jnp.zeros((1, L*L)))

# Metropolis Local Sampling
N_samples = 4096
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=0.03)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=N_samples,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)

# Variational monte carlo driver
vmc = nk.driver.VMC_SR(hamiltonian=hamiltonian,
            optimizer=optimizer,
            diag_shift=1e-4,
            variational_state=vstate,
            mode='complex',
            chunk_size_bwd=N_samples,
            on_the_fly=True,
            use_ntk=True)

log = nk.logging.RuntimeLog()

N_opt = 800
vmc.run(n_iter=N_opt, out=log)