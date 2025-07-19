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
# # Part 2: Variational States with Full Summation
#
# In this second tutorial, we will:
# - Implement variational ansätze using Flax
# - Compute energies using full summation over the Hilbert space
# - Learn JAX/JIT compilation techniques
# - Implement gradient computation and optimization
# - Explore different variational ansätze (Mean Field and Jastrow)
#
# This tutorial builds on the Hamiltonian and operator concepts from Part 1.

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

# jax and jax.numpy
import jax
import jax.numpy as jnp

# Flax for neural network models
import flax.linen as nn

print("Python version (requires >=3.9)", platform.python_version())
print("NetKet version (requires >=3.9.1)", nk.__version__)

# %% [markdown]
# ## 1. Setup from Previous Tutorial
#
# Let's quickly recreate the system from Part 1:

# %%
# Define the system
L = 4
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)
hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)

# Build the Hamiltonian (solution from Part 1)
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
# ## 2. Variational Ansatz & JAX/Flax Fundamentals
#
# In this section, we'll implement variational ansätze to approximate the ground state. We'll use JAX and Flax to define models that compute the **logarithm** of the wave-function amplitudes.
#
# For a variational state $|\Psi\rangle$, we define:
#
# $$ \langle \sigma^{z}_1,\dots \sigma^{z}_N| \Psi \rangle = \exp\left[\mathrm{Model}(\sigma^{z}_1,\dots \sigma^{z}_N ; \theta ) \right], $$
#
# where $\theta$ are the variational parameters.

# %% [markdown]
# ### 2.1 Mean-Field Ansatz
#
# We now would like to find a variational approximation of the ground state of this Hamiltonian. 
# As a first step, we can try to use a very simple mean field ansatz: 
#
# $$ \langle \sigma^{z}_1,\dots \sigma^{z}_N| \Psi_{\mathrm{mf}} \rangle = \Pi_{i=1}^{N} \Phi(\sigma^{z}_i), $$
#
# where the variational parameters are the single-spin wave functions, which we can further take to be normalized: 
#
# $$ |\Phi(\uparrow)|^2 + |\Phi(\downarrow)|^2 =1, $$
#
# and we can further write $ \Phi(\sigma^z) = \sqrt{P(\sigma^z)}e^{i \phi(\sigma^z)}$. In order to simplify the presentation, we take here and in the following examples the phase $ \phi=0 $. In this specific model this is without loss of generality, since it is known that the ground state is real and positive. 
#
# For the normalized single-spin probability we will take a sigmoid form: 
#
# $$ P(\sigma_z; \lambda) = 1/(1+\exp(-\lambda \sigma_z)), $$
#
# thus depending on the real-valued variational parameter $\lambda$. 
# In NetKet one has to define a variational function approximating the **logarithm** of the wave-function amplitudes (or density-matrix values).
# We call this variational function _the Model_ (yes, caps on the M).
#
# $$ \langle \sigma^{z}_1,\dots \sigma^{z}_N| \Psi_{\mathrm{mf}} \rangle = \exp\left[\mathrm{Model}(\sigma^{z}_1,\dots \sigma^{z}_N ; \theta ) \right], $$
#
# where $\theta$ is a set of parameters. 
# In this case, the parameter of the model will be just one: $\lambda$.  
#
# The Model can be defined using one of the several *functional* jax frameworks such as Jax/Stax, Flax or Haiku. 
# NetKet includes several pre-built models and layers built with [Flax](https://github.com/google/flax), so we will be using it for the rest of the notebook. 


# %%
# A Flax model must be a class subclassing `nn.Module`
class MF(nn.Module):

    # The __call__(self, x) function should take as
    # input a batch of states x.shape = (n_samples, N)
    # and should return a vector of n_samples log-amplitudes
    @nn.compact
    def __call__(self, x):

        # A tensor of variational parameters is defined by calling
        # the method `self.param` where the arguments are:
        # - arbitrary name used to refer to this set of parameters
        # - an initializer used to provide the initial values.
        # - The shape of the tensor
        # - The dtype of the tensor.
        lam = self.param("lambda", nn.initializers.normal(), (1,), float)

        # compute the probabilities
        p = nn.log_sigmoid(lam * x)

        # sum the output
        return 0.5 * jnp.sum(p, axis=-1)


# %% [markdown]
# ### 2.2 Working with Flax Models
#
# The model itself is only a set of instructions. To initialize parameters:

# %%
# create an instance of the model
model = MF()

# pick a RNG key to initialise the random parameters
key = jax.random.key(0)

# initialise the weights
parameters = model.init(key, np.random.rand(hi.size))

print("Parameters structure:")
print(parameters)

# %% [markdown]
# Parameters are stored as 'pytrees' - nested dictionaries whose leaves are numerical arrays. You can apply mathematical operations using `jax.tree.map`:

# %%
# Examples of tree operations
dict1 = {"a": 1, "b": 2}
dict2 = {"a": 1, "b": -2}

print("multiply_by_10:             ", jax.tree.map(lambda x: 10 * x, dict1))
print("add dict1 and dict2:       ", jax.tree.map(lambda x, y: x + y, dict1, dict2))
print("subtract dict1 and dict2:  ", jax.tree.map(lambda x, y: x - y, dict1, dict2))

# %% [markdown]
# To evaluate the model:

# %%
# generate 4 random inputs
inputs = hi.random_state(jax.random.key(1), (4,))

log_psi = model.apply(parameters, inputs)
print(f"Log-psi shape: {log_psi.shape}")
print(f"Log-psi values: {log_psi}")

# %% [markdown]
# ## 3. Exercise: Converting to Normalized Wavefunction
#
# Write a function that takes the model and parameters, and returns the exponentiated wavefunction, properly normalized.


# %% tags=["placeholder", "hide-input", "hide-output"]
def to_array(model, parameters):
    # Begin by generating all configurations in the hilbert space.
    all_configurations = hi.all_states()

    # TODO: Evaluate the model and convert to a normalised wavefunction.
    # Hint: Use model.apply, jnp.exp, and jnp.linalg.norm

    return None  # TODO: return normalized wavefunction


# %% tags=["solution"]
def to_array(model, parameters):
    # begin by generating all configurations in the hilbert space.
    all_configurations = hi.all_states()

    # now evaluate the model, and convert to a normalised wavefunction.
    logpsi = model.apply(parameters, all_configurations)
    psi = jnp.exp(logpsi)
    psi = psi / jnp.linalg.norm(psi)
    return psi

# %% [markdown]
# Test your implementation:

# %%
# Uncomment after implementing to_array
# assert to_array(model, parameters).shape == (hi.n_states, )
# assert np.all(to_array(model, parameters) > 0)
# np.testing.assert_allclose(np.linalg.norm(to_array(model, parameters)), 1.0)
# print("to_array implementation is correct!")

# %% [markdown]
# ### 3.1 JAX JIT Compilation
#
# If you implemented everything using `jnp.` operations, we can JIT-compile for speed:

# %%
# Uncomment after implementing to_array
# static_argnames must be used for any argument that is not a pytree or array
# to_array_jit = jax.jit(to_array, static_argnames="model")

# Run once to compile
# to_array_jit(model, parameters)
# print("JIT compilation successful!")

# %% [markdown]
# ## 4. Exercise: Computing Energy
#
# Write a function that computes the energy of the variational state:


# %% tags=["placeholder", "hide-input", "hide-output"]
def compute_energy(model, parameters, hamiltonian_sparse):
    # TODO: Get the wavefunction and compute <psi|H|psi>
    # Hint: Use to_array and matrix multiplication

    return None  # TODO: return energy


# %% tags=["solution"]
def compute_energy(model, parameters, hamiltonian_sparse):
    psi = to_array(model, parameters)
    return psi.conj().T @ (hamiltonian_sparse @ psi)

# %%
# Test your implementation
hamiltonian_sparse = hamiltonian.to_sparse()
hamiltonian_jax_sparse = hamiltonian_jax.to_sparse()

assert compute_energy(model, parameters, hamiltonian_sparse).shape == ()
assert compute_energy(model, parameters, hamiltonian_sparse) < 0
print("compute_energy implementation is correct!")

# We can also JIT-compile this
# compute_energy_jit = jax.jit(compute_energy, static_argnames="model")

# %% [markdown]
# ## 5. Gradient Computation
#
# JAX makes computing gradients easy. We can differentiate the energy with respect to parameters:

# %%
from functools import partial


# JIT the combined energy and gradient function
@partial(jax.jit, static_argnames="model")
def compute_energy_and_gradient(model, parameters, hamiltonian_sparse):
    grad_fun = jax.value_and_grad(compute_energy, argnums=1)
    return grad_fun(model, parameters, hamiltonian_sparse)


# Example usage (uncomment after implementing compute_energy)
# energy, gradient = compute_energy_and_gradient(model, parameters, hamiltonian_jax_sparse)
# print("Energy:", energy)
# print("Gradient structure:", jax.tree.map(lambda x: x.shape, gradient))

# %% [markdown]
# ## 6. Exercise: Optimization Loop
#
# Now implement an optimization loop to find the ground state. Use gradient descent with learning rate 0.01 for 100 iterations:

# %% tags=["placeholder", "hide-input", "hide-output"]
from tqdm.auto import tqdm

# Initialize
model = MF()
parameters = model.init(jax.random.key(0), np.ones((hi.size,)))

# Logging
logger = nk.logging.RuntimeLog()

iterations = 100

for i in tqdm(range(iterations)):
    # TODO: compute energy and gradient
    # energy, gradient = ...

    # TODO: update parameters using gradient descent
    # parameters = jax.tree.map(lambda x,y: x - 0.01*y, parameters, gradient)

    # Log energy
    # logger(step=i, item={'Energy': energy})
    pass

# %% tags=["solution"]
from tqdm.auto import tqdm

# Initialize
model = MF()
parameters = model.init(jax.random.key(0), np.ones((hi.size, )))

# Logging
logger = nk.logging.RuntimeLog()

for i in tqdm(range(100)):
    # compute energy and gradient
    energy, gradient = compute_energy_and_gradient(model, parameters, hamiltonian_jax_sparse)

    # update parameters
    parameters = jax.tree.map(lambda x,y:x-0.01*y, parameters, gradient)

    # log energy
    logger(step=i, item={'Energy':energy})

# %% [markdown]
# Plot the optimization progress:

# %%
# Uncomment after running optimization
# plt.figure(figsize=(10, 6))
# plt.subplot(1, 2, 1)
# plt.plot(logger.data['Energy']['iters'], logger.data['Energy']['value'])
# plt.xlabel('Iteration')
# plt.ylabel('Energy')
# plt.title('Energy vs Iteration')

# plt.subplot(1, 2, 2)
# plt.semilogy(logger.data['Energy']['iters'], np.abs(logger.data['Energy']['value'] - e_gs))
# plt.xlabel('Iteration')
# plt.ylabel('|Energy - Exact|')
# plt.title('Error vs Iteration (log scale)')
# plt.tight_layout()

# %% [markdown]
# ## 7. Exercise: Jastrow Ansatz
#
# Now implement a more sophisticated Jastrow ansatz:
#
# $$ \langle \sigma^{z}_1,\dots \sigma^{z}_N| \Psi_{\mathrm{jas}} \rangle = \exp \left( \sum_{ij}\sigma_i J_{ij}\sigma_j\right),$$


# %% tags=["placeholder", "hide-input", "hide-output"]
class Jastrow(nn.Module):

    @nn.compact
    def __call__(self, input_x):

        n_sites = input_x.shape[-1]

        # Define the J matrix parameter
        J = self.param("J", nn.initializers.normal(), (n_sites, n_sites), float)

        # Ensure same data types
        dtype = jax.numpy.promote_types(J.dtype, input_x.dtype)
        J = J.astype(dtype)
        input_x = input_x.astype(dtype)

        # Symmetrize J matrix
        J_symm = J.T + J

        # TODO: Compute the result using vectorized operations
        # Hint: use jnp.einsum("...i,ij,...j", input_x, J_symm, input_x)
        res = None # TODO: implement this

        return res


# %% tags=["solution"]
class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, x):
        n_sites = x.shape[-1]
        J = self.param(
            "J", nn.initializers.normal(), (n_sites,n_sites), float
        )
        J_symm = J.T + J
        return jnp.einsum("...i,ij,...j", x, J_symm, x)

# %% [markdown]
# Test the Jastrow implementation:

# %%
# Uncomment after implementing Jastrow
# model_jastrow = Jastrow()

# one_sample = hi.random_state(jax.random.key(0))
# batch_samples = hi.random_state(jax.random.key(0), (5,))
# multibatch_samples = hi.random_state(jax.random.key(0), (5,4,))

# parameters_jastrow = model_jastrow.init(jax.random.key(0), one_sample)
# assert parameters_jastrow['params']['J'].shape == (hi.size, hi.size)
# assert model_jastrow.apply(parameters_jastrow, one_sample).shape == ()
# assert model_jastrow.apply(parameters_jastrow, batch_samples).shape == batch_samples.shape[:-1]
# assert model_jastrow.apply(parameters_jastrow, multibatch_samples).shape == multibatch_samples.shape[:-1]
# print("Jastrow implementation is correct!")

# %% [markdown]
# ## 8. Exercise: Optimize the Jastrow Ansatz
#
# Repeat the optimization analysis with the Jastrow ansatz and compare the results:

# %% tags=["placeholder", "hide-input", "hide-output"]
# TODO: Implement optimization loop for Jastrow ansatz
# Use the same structure as before but with model_jastrow

# %% [markdown]
# ## Summary
#
# In this tutorial, you learned:
# - How to implement variational ansätze using JAX and Flax
# - How to compute energies using full summation over the Hilbert space
# - How to use JAX for automatic differentiation and JIT compilation
# - How to implement optimization loops for variational parameters
# - How to compare different ansätze (Mean Field vs Jastrow)
#
# In the next tutorial, we will extend this to Monte Carlo sampling for larger systems where full summation is not feasible.

# %%
