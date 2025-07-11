# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Part 3: Monte Carlo Sampling
#
# In this final tutorial, we will:
# - Implement Monte Carlo sampling for larger systems
# - Compute local energies using sparse operator connections
# - Estimate energies and gradients from samples
# - Build a complete VMC optimization loop
# - Explore advanced optimizers and future extensions
#
# This tutorial builds on the concepts from Parts 1 and 2, extending to the Monte Carlo regime where full summation over the Hilbert space is no longer feasible.

# %% [markdown]
# :::{note}
# If you are executing this notebook on **Colab**, you will need to install NetKet:
# :::

# %% mystnb={"code_prompt_hide": "Hide", "code_prompt_show": "Show and uncomment to install NetKet on colab"}
# # %pip install --quiet netket

# %% tags=["hide-cell"]
# Import necessary libraries
import platform
import netket as nk
import numpy as np
from tqdm.auto import tqdm
from functools import partial

# jax and jax.numpy
import jax
import jax.numpy as jnp

# Flax for neural network models
import flax.linen as nn

print("Python version (requires >=3.9)", platform.python_version())
print("NetKet version (requires >=3.9.1)", nk.__version__)

# %% [markdown]
# ## 1. Setup from Previous Tutorials
#
# Let's recreate the complete system from Parts 1 and 2:

# %%
# Define the system
L = 4
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Build the Hamiltonian
hamiltonian = nk.operator.LocalOperator(hi)

# Add transverse field terms
for site in g.nodes():
    hamiltonian = hamiltonian - 1.0 * nk.operator.spin.sigmax(hi, site)

# Add Ising interaction terms
for i, j in g.edges():
    hamiltonian = hamiltonian + nk.operator.spin.sigmaz(
        hi, i
    ) @ nk.operator.spin.sigmaz(hi, j)

# Convert to JAX format
hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()

# Compute exact ground state for comparison
from scipy.sparse.linalg import eigsh

e_gs, psi_gs = eigsh(hamiltonian.to_sparse(), k=1)
e_gs = e_gs[0]
psi_gs = psi_gs.reshape(-1)

print(f"Exact ground state energy: {e_gs:.6f}")

# %% [markdown]
# ## 2. Variational Models from Part 2
#
# Let's redefine our variational ansätze:


# %%
# Mean Field Ansatz
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)
        p = nn.log_sigmoid(lam * x)
        return 0.5 * jnp.sum(p, axis=-1)


# Jastrow Ansatz
class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        J = self.param("J", nn.initializers.normal(), (n_sites, n_sites), float)
        dtype = jax.numpy.promote_types(J.dtype, x.dtype)
        J = J.astype(dtype)
        x = x.astype(dtype)
        J_symm = J.T + J
        return jnp.einsum("...i,ij,...j", x, J_symm, x)


# %% [markdown]
# ## 3. Monte Carlo Sampling
#
# For larger problems, we cannot sum over the whole Hilbert space. Instead, we use Monte Carlo sampling to generate configurations according to $|\psi(\sigma)|^2$.

# %% [markdown]
# ### 3.1 Setting up the Sampler
#
# We use a Metropolis sampler that proposes new states by flipping individual spins:

# %%
sampler = nk.sampler.MetropolisSampler(
    hi,  # the hilbert space to be sampled
    nk.sampler.rules.LocalRule(),  # the transition rule
    n_chains=20,  # number of parallel chains
)

# %% [markdown]
# ### 3.2 Generating Samples
#
# Samplers are used as follows:
# 1. Initialize the sampler state
# 2. Reset when changing parameters
# 3. Call `sample` to generate new configurations

# %%
# Example with Mean Field model
model = MF()
parameters = model.init(jax.random.key(0), np.ones((hi.size,)))

# Initialize sampler state
sampler_state = sampler.init_state(model, parameters, seed=1)
sampler_state = sampler.reset(model, parameters, sampler_state)

# Generate samples
samples, sampler_state = sampler.sample(
    model, parameters, state=sampler_state, chain_length=100
)

print(f"Sample shape: {samples.shape}")
# Dimensions: (n_chains, chain_length, n_sites)
# Note: chains are sometimes referred to as walkers

# %% [markdown]
# ## 4. Computing Local Energies
#
# We want to compute the energy as an expectation value:
#
# $$
#    E = \sum_i^{N_s} \frac{E_\text{loc}(\sigma_i)}{N_s}
# $$
#
# where $\sigma_i$ are the samples and $E_\text{loc}$ is the local energy:
#
# $$
#   E_\text{loc}(\sigma) = \frac{\langle \sigma |H|\psi\rangle}{\langle \sigma |\psi\rangle} = \sum_\eta \langle\sigma|H|\eta\rangle \frac{\psi(\eta)}{\psi(\sigma)}
# $$

# %% [markdown]
# ### 4.1 Understanding Operator Connections
#
# The sum over $\eta$ is only over configurations connected to $\sigma$ by the Hamiltonian (i.e., where $\langle\sigma|H|\eta\rangle \neq 0$). NetKet's operators provide this efficiently:

# %%
# Example: get connections for a single configuration
sigma = hi.random_state(jax.random.key(1))
eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

print(f"Input configuration shape: {sigma.shape}")
print(f"Connected configurations shape: {eta.shape}")
print(f"Matrix elements shape: {H_sigmaeta.shape}")

# For this Hamiltonian, each site connects to itself (diagonal) and its neighbors

# %% [markdown]
# This also works for batches of configurations:

# %%
sigma_batch = hi.random_state(jax.random.key(1), (4, 5))
eta_batch, H_batch = hamiltonian_jax.get_conn_padded(sigma_batch)

print(f"Batch input shape: {sigma_batch.shape}")
print(f"Batch connected configurations shape: {eta_batch.shape}")
print(f"Batch matrix elements shape: {H_batch.shape}")

# %% [markdown]
# ## 5. Exercise: Computing Local Energies
#
# Implement a function to compute local energies using the connection information:
#
# $$
#   E_\text{loc}(\sigma) = \sum_\eta \langle\sigma|H|\eta\rangle \exp[\log\psi(\eta) - \log\psi(\sigma)]
# $$


# %% tags=["placeholder", "hide-input", "hide-output"]
def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

    # TODO: Compute log-psi for original configurations
    logpsi_sigma = None  # model.apply(parameters, sigma)

    # TODO: Compute log-psi for connected configurations
    logpsi_eta = None  # model.apply(parameters, eta)

    # TODO: To match dimensions for broadcasting, expand logpsi_sigma
    # Hint: jnp.expand_dims(logpsi_sigma, -1) might help

    # TODO: Compute local energies
    # res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    return None  # TODO


# %% tags=["solution"]
def compute_local_energies(model, parameters, hamiltonian_jax, sigma):
    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)

    logpsi_sigma = model.apply(parameters, sigma)
    logpsi_eta = model.apply(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)

    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis=-1)

    return res

# %% [markdown]
# Test your implementation:

# %%
# Uncomment after implementing compute_local_energies
# assert compute_local_energies(model, parameters, hamiltonian_jax, samples[0]).shape == samples.shape[1:-1]
# assert compute_local_energies(model, parameters, hamiltonian_jax, samples).shape == samples.shape[:-1]

# Check that it JIT compiles
# jax.jit(compute_local_energies, static_argnames='model')(model, parameters, hamiltonian_jax, sigma)
# print("compute_local_energies implementation is correct!")

# %% [markdown]
# ## 6. Exercise: Estimating Energy from Samples
#
# Write a function that estimates the energy and its statistical error from samples. The error is given by:
#
# $$
#     \epsilon_E = \sqrt{\frac{\mathbb{V}\text{ar}(E_\text{loc})}{N_\text{samples}}}
# $$


# %% tags=["placeholder", "hide-input", "hide-output"]
@partial(jax.jit, static_argnames="model")
def estimate_energy(model, parameters, hamiltonian_jax, sigma):
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)

    # TODO: Compute average energy
    E_average = None  # jnp.mean(E_loc)

    # TODO: Compute variance
    E_variance = None  # jnp.var(E_loc)

    # TODO: Compute error of the mean
    E_error = None  # jnp.sqrt(E_variance / E_loc.size)

    # Return a NetKet Stats object
    return nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)


# %% tags=["solution"]
@partial(jax.jit, static_argnames='model')
def estimate_energy(model, parameters, hamiltonian_jax, sigma):
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)

    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)

    return nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)

# %% [markdown]
# Test the energy estimation:

# %%
# Uncomment after implementing estimate_energy
# energy_estimate = estimate_energy(model, parameters, hamiltonian_jax, samples)
# print("Energy estimate:", energy_estimate)

# %% [markdown]
# Let's verify our Monte Carlo estimate against the exact calculation by generating more samples:

# %%
# Uncomment after implementing functions
# samples_many, sampler_state = sampler.sample(model, parameters, state=sampler_state, chain_length=5000)

# Compare with full summation from Part 2
# def compute_energy_exact(model, parameters, hamiltonian_sparse):
#     all_configurations = hi.all_states()
#     logpsi = model.apply(parameters, all_configurations)
#     psi = jnp.exp(logpsi)
#     psi = psi / jnp.linalg.norm(psi)
#     return psi.conj().T @ (hamiltonian_sparse @ psi)

# hamiltonian_sparse = hamiltonian.to_sparse()
# exact_energy = compute_energy_exact(model, parameters, hamiltonian_sparse)
# mc_estimate = estimate_energy(model, parameters, hamiltonian_jax, samples_many)

# print(f"Exact calculation: {exact_energy:.6f}")
# print(f"MC estimate: {mc_estimate}")

# %% [markdown]
# ## 7. Gradient Estimation with Monte Carlo
#
# The gradient of the energy can be estimated using:
#
# $$
#     \nabla_k E = \mathbb{E}_{\sigma\sim|\psi(\sigma)|^2} \left[ (\nabla_k \log\psi(\sigma))^* \left( E_\text{loc}(\sigma) - \langle E \rangle\right)\right]
# $$
#
# We can compute this efficiently using JAX's vector-Jacobian product (VJP).

# %% [markdown]
# ### 7.1 Understanding the Jacobian
#
# Think of $\nabla_k \log\psi(\sigma_i)$ as the JACOBIAN of the function $\log\psi_\sigma : \mathbb{R}^{N_\text{pars}} \rightarrow \mathbb{R}^{N_\text{samples}}$:

# %%
# Example with Jastrow model
model_jastrow = Jastrow()
parameters_jastrow = model_jastrow.init(
    jax.random.key(0), hi.random_state(jax.random.key(0))
)

# Reshape samples to a vector
sigma_vector = samples.reshape(-1, hi.size)

# Define the function to differentiate
logpsi_sigma_fun = lambda pars: model_jastrow.apply(pars, sigma_vector)

print(f"Input parameters shape: {jax.tree.map(lambda x: x.shape, parameters_jastrow)}")
print(f"Output shape: {logpsi_sigma_fun(parameters_jastrow).shape}")

# We can compute the Jacobian
jacobian = jax.jacrev(logpsi_sigma_fun)(parameters_jastrow)
print(f"Jacobian shape: {jax.tree.map(lambda x: x.shape, jacobian)}")

# %% [markdown]
# ## 8. Exercise: Energy and Gradient Estimation
#
# Implement a function that computes both energy and gradient estimates using VJP:


# %% tags=["placeholder", "hide-input", "hide-output"]
@partial(jax.jit, static_argnames="model")
def estimate_energy_and_gradient(model, parameters, hamiltonian_jax, sigma):
    # Reshape samples to remove extra batch dimensions
    sigma = sigma.reshape(-1, sigma.shape[-1])

    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)

    # TODO: Compute energy statistics
    E_average = None  # jnp.mean(E_loc)
    E_variance = None  # jnp.var(E_loc)
    E_error = None  # jnp.sqrt(E_variance/E_loc.size)
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)

    # TODO: Compute gradient using VJP
    # Define function to differentiate
    logpsi_sigma_fun = lambda pars: model.apply(pars, sigma)

    # Use VJP to compute gradient efficiently
    # _, vjpfun = jax.vjp(logpsi_sigma_fun, parameters)
    # E_grad = vjpfun((E_loc - E_average)/E_loc.size)[0]

    return E, None  # E_grad


# %% tags=["solution"]
@partial(jax.jit, static_argnames='model')
def estimate_energy_and_gradient(model, parameters, hamiltonian_jax, sigma):
    # reshape the samples to a vector of samples with no extra batch dimensions
    sigma = sigma.reshape(-1, sigma.shape[-1])

    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)

    # compute the energy as well
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance/E_loc.size)
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)

    # compute the gradient using VJP
    logpsi_sigma_fun = lambda pars : model.apply(pars, sigma)
    _, vjpfun = jax.vjp(logpsi_sigma_fun, parameters)
    E_grad = vjpfun((E_loc - E_average)/E_loc.size)[0]

    return E, E_grad

# %% [markdown]
# ## 9. Exercise: Complete VMC Optimization Loop
#
# Now implement a complete VMC optimization using Monte Carlo sampling:

# %% tags=["placeholder", "hide-input", "hide-output"]
# Settings
model = MF()  # Try both MF() and Jastrow()
sampler = nk.sampler.MetropolisSampler(hi, nk.sampler.rules.LocalRule(), n_chains=20)
n_iters = 300
chain_length = 1000 // sampler.n_chains

# Initialize
parameters = model.init(jax.random.key(0), np.ones((hi.size,)))
sampler_state = sampler.init_state(model, parameters, seed=1)

# Logging
logger = nk.logging.RuntimeLog()

for i in tqdm(range(n_iters)):
    # TODO: Sample configurations
    # sampler_state = sampler.reset(model, parameters, state=sampler_state)
    # samples, sampler_state = sampler.sample(model, parameters, state=sampler_state, chain_length=chain_length)

    # TODO: Compute energy and gradient
    # E, E_grad = estimate_energy_and_gradient(model, parameters, hamiltonian_jax, samples)

    # TODO: Update parameters using gradient descent (learning rate ~0.005)
    # parameters = jax.tree.map(lambda x,y: x-0.005*y, parameters, E_grad)

    # TODO: Log energy
    # logger(step=i, item={'Energy': E})
    pass

# %% tags=["solution"]
# Settings
model = Jastrow()  # Try both MF() and Jastrow()
sampler = nk.sampler.MetropolisSampler(
    hi,
    nk.sampler.rules.LocalRule(),
    n_chains=20
)
n_iters = 300
chain_length = 1000 // sampler.n_chains

# Initialize
parameters = model.init(jax.random.key(0), np.ones((hi.size,)))
sampler_state = sampler.init_state(model, parameters, seed=1)

# Logging
logger = nk.logging.RuntimeLog()

for i in tqdm(range(n_iters)):
    # sample
    sampler_state = sampler.reset(model, parameters, state=sampler_state)
    samples, sampler_state = sampler.sample(model, parameters, state=sampler_state, chain_length=chain_length)

    # compute energy and gradient
    E, E_grad = estimate_energy_and_gradient(model, parameters, hamiltonian_jax, samples)

    # update parameters
    parameters = jax.tree.map(lambda x,y: x-0.005*y, parameters, E_grad)

    # log energy
    logger(step=i, item={'Energy':E})

# %% [markdown]
# Plot the optimization results:

# %%
# Uncomment after running optimization
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(logger.data['Energy']['iters'], logger.data['Energy']['Mean'])
# plt.axhline(y=e_gs, color='r', linestyle='--', label='Exact ground state')
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title('VMC Energy vs Iteration')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.semilogy(logger.data['Energy']['iters'], np.abs(logger.data['Energy']['Mean'] - e_gs))
# plt.xlabel('Iteration')
# plt.ylabel('|Energy - Exact|')
# plt.title('Error vs Iteration (log scale)')
# plt.tight_layout()

# %% [markdown]
# ## 10. Advanced Topics and Extensions

# %% [markdown]
# ### 10.1 Better Optimizers with Optax
#
# You can use more sophisticated optimizers from the [optax](https://optax.readthedocs.io/en/latest/) library:

# %%
import optax


# Example optimization loop with Adam
def optimize_with_adam():
    # Define optimizer
    optimizer = optax.adam(learning_rate=0.01)

    # Initialize
    model = Jastrow()
    parameters = model.init(jax.random.key(0), np.ones((hi.size,)))
    optimizer_state = optimizer.init(parameters)

    logger = nk.logging.RuntimeLog()

    for i in tqdm(range(100)):
        # Sample and compute gradients (same as before)
        # samples, sampler_state = ...
        # E, E_grad = estimate_energy_and_gradient(...)

        # Update with Adam
        # updates, optimizer_state = optimizer.update(E_grad, optimizer_state, parameters)
        # parameters = optax.apply_updates(parameters, updates)

        # logger(step=i, item={'Energy': E})
        pass


# %% [markdown]
# ### 10.2 Feed-Forward Neural Networks
#
# Try implementing a more complex ansatz using feed-forward networks:


# %% tags=["placeholder", "hide-input", "hide-output"]
class FeedForward(nn.Module):
    hidden_size: int = 32

    @nn.compact
    def __call__(self, x):
        # TODO: Implement a 2-layer feed-forward network
        # Use nn.Dense layers with relu activation
        # Example structure:
        # x = nn.Dense(self.hidden_size)#...
        # x = nn.relu(x)
        # x = nn.Dense(self.hidden_size)#...
        # return jnp.sum(x, axis=-1)  # Pool over sites
        pass

# %% tags=["solution"]
class FeedForward(nn.Module):
    hidden_size: int = 32

    @nn.compact
    def __call__(self, x):
        # Use nn.Dense layers with relu activation
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)  # Pool over sites


# %% [markdown]
# ### 10.3 Comparison of Different Ansätze
#
# Compare the performance of different variational ansätze:


# %% tags=["placeholder", "hide-input", "hide-output"]
def compare_ansatze():
    """Compare Mean Field, Jastrow, and Feed-Forward ansätze"""
    results = {}

    for name, model_class in [("MeanField", MF), ("Jastrow", Jastrow)]:
        print(f"Optimizing {name} ansatz...")

        # Run optimization (implement the loop)
        # Store final energy in results[name]

    # Plot comparison
    # plt.figure()
    # for name, energy_history in results.items():
    #     plt.semilogy(energy_history - e_gs, label=name)
    # plt.xlabel('Iteration')
    # plt.ylabel('Energy Error')
    # plt.legend()
    # plt.title('Comparison of Variational Ansätze')


# %% [markdown]
# ## Summary
#
# In this tutorial, you learned:
# - How to implement Monte Carlo sampling for VMC calculations
# - How to compute local energies using operator connections
# - How to estimate energies and gradients from samples
# - How to build complete VMC optimization loops
# - How to use advanced optimizers and neural network architectures
#
# You now have the tools to quickly get started in running VMC calculations without worrying about the implementation details of sampling and operators. This provides a foundation for implementing more advanced techniques like:
#
# - Stochastic Reconfiguration (Natural Gradients)
# - Time evolution and dynamics
# - More sophisticated neural network architectures
# - Multi-GPU distributed calculations
#
# The modular design allows you to focus on the physics and machine learning aspects while NetKet handles the computational infrastructure.

# %%
