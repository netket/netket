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
#     display_name: Python (Netket development)
#     language: python
#     name: dev-netket
# ---

# %% [markdown]
# # Ground-State: J1-J2 model
#
# In this tutorial we will use NetKet to obtain the ground state of the J1-J2 model in one-dimension with periodic boundary conditions, using a Neural Network variational wave-function. The Hamiltonian of the model is given by:
#
# $$ H = \sum_{i=1}^{L} J_{1}\vec{\sigma}_{i} \cdot \vec{\sigma}_{i+1} + J_{2} \vec{\sigma}_{i} \cdot \vec{\sigma}_{i+2} $$
#
# where the sum is over sites of the 1-D chain. Here $\vec{\sigma}=(\sigma^x,\sigma^y,\sigma^z)$ is the vector of Pauli matrices.
#
# We will also explore some useful functionalities provided by the package.
#
# ## Objectives:
#     1. Defining custom Hamiltonians
#     2. Defining the machine (variational ansatz)
#     3. Variational Monte Carlo Optimisation
#     4. Measuring observables
#     5. Data Visualisation
#     6. Sanity Check: Exact Diagonalisation
#
# Let's start.

# %%
# ensure we run on the CPU
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

# Import netket library
import netket as nk

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ## 1) Defining a Custom Hamiltonian
#
# The first thing to do is to define the graph (lattice) on which to specify the Hamiltonian. Here we would like to build a one-dimensional graph with both nearest and next nearest neighbour bonds. The graph is created in the ``nk.graph.CustomGraph`` class. To initialise the class we simply provide a list of edges in the ``[[site_i, site_j, edge_color], ...]``

# %%
# Couplings J1 and J2
J = [1, 0.2]
L = 14

# %%
# Define custom graph
edge_colors = []
for i in range(L):
    edge_colors.append([i, (i + 1) % L, 1])
    edge_colors.append([i, (i + 2) % L, 2])

# Define the netket graph object
g = nk.graph.Graph(edges=edge_colors)

# %% [markdown]
# We specify a different ``color`` for each type of bond so as to define a different operator for each of them.
# Next, we define the relevant bond operators.

# %%
# Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = np.kron(sigmaz, sigmaz)

# Exchange interactions
exchange = np.asarray([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])

bond_operator = [
    (J[0] * mszsz).tolist(),
    (J[1] * mszsz).tolist(),
    (-J[0] * exchange).tolist(),
    (J[1] * exchange).tolist(),
]

bond_color = [1, 2, 1, 2]

# %% [markdown]
# **Side Remark**: Notice the minus sign in front of the exchange. This is simply a basis rotation corresponding to the Marshall sign rule (as an exercise, change the sign of this exchange and observe that the exact diagonalization results in Section 6 do not change). The goal of this basis rotation is to speed up the convergence of the Monte Carlo simulations of the wave-function (by providing a good variational sign structure to start with), but in principle the converged results should be identical in both bases. Note further that this sign change is useful at low frustration (such as here $J_2=0.2$), but may actually be not optimal at stronger frustration. As a bonus exercise, repeat the calculation with $J_2=0.8$, and see which basis (*i.e.* which sign in front of the exchange) leads to faster convergence.

# %% [markdown]
# Before defining the Hamiltonian, we also need to specify the Hilbert space. For our case, this would be the chain spin-half degrees of freedom.

# %%
# Spin based Hilbert Space
hi = nk.hilbert.Spin(s=0.5, total_sz=0.0, N=g.n_nodes)

# %% [markdown]
# Note that we impose here the total magnetization to be zero (it turns out to be the correct magnetization for the ground-state). As an exercise, check that the energy of the lowest state in other magnetization sectors is larger.
#
# Next, we define the custom graph Hamiltonian using the ``nk.operator.GraphOperator`` class, by providing the hilbert space ``hi``, the bond operators ``bondops=bond_operator`` and the corresponding bond color ``bondops_colors=bond_color``. The information about the graph (bonds and bond colors) are contained within the ``nk.hilbert.Spin`` object ``hi``.

# %%
# Custom Hamiltonian operator
op = nk.operator.GraphOperator(
    hi, graph=g, bond_ops=bond_operator, bond_ops_colors=bond_color
)

# %% [markdown]
# ## 2) Defining the Machine
#
# For this tutorial, we shall use the most common type of neural network: fully connected feedforward neural network ``nk.machine.FFNN``. Other types of neural networks available will be discussed in other tutorials.

# %%
import netket.nn as nknn
import flax.linen as nn

import jax.numpy as jnp


class FFNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            features=2 * x.shape[-1],
            use_bias=True,
            param_dtype=np.complex128,
            kernel_init=nn.initializers.normal(stddev=0.01),
            bias_init=nn.initializers.normal(stddev=0.01),
        )(x)
        x = nknn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x


model = FFNN()

# %% [markdown]
# ## 3) Variational Monte Carlo Optimisation
#
# We have now set up our model (Hamiltonian, Graph, Hilbert Space) and can proceed to optimise the variational ansatz we chose, namely the ``ffnn`` machine.
#
# To setup the variational Monte Carlo optimisation tool, we have to provide a sampler ``nk.sampler`` and an optimizer ``nk.optimizer``.

# %%
# We shall use an exchange Sampler which preserves the global magnetization (as this is a conserved quantity in the model)
sa = nk.sampler.MetropolisExchange(hilbert=hi, graph=g, d_max=2)

# Construct the variational state
vs = nk.vqs.MCState(sa, model, n_samples=1008)

# We choose a basic, albeit important, Optimizer: the Stochastic Gradient Descent
opt = nk.optimizer.Sgd(learning_rate=0.01)

# Stochastic Reconfiguration
sr = nk.optimizer.SR(diag_shift=0.01)

# We can then specify a Variational Monte Carlo object, using the Hamiltonian, sampler and optimizers chosen.
# Note that we also specify the method to learn the parameters of the wave-function: here we choose the efficient
# Stochastic reconfiguration (Sr), here in an iterative setup
gs = nk.VMC(hamiltonian=op, optimizer=opt, variational_state=vs, preconditioner=sr)

# %% [markdown]
# ## 4) Measuring Observables
#
# Before running the optimization, it can be helpful to add some observables to keep track off during the optimization. For our purpose, let us measure the antiferromagnetic structure factor, defined as:
#
# $$ \frac{1}{L} \sum_{ij} \langle \hat{\sigma}_{i}^z \cdot \hat{\sigma}_{j}^z\rangle e^{i\pi(i-j)}$$.

# %%
# We need to specify the local operators as a matrix acting on a local Hilbert space
sf = []
sites = []
structure_factor = nk.operator.LocalOperator(hi, dtype=complex)
for i in range(0, L):
    for j in range(0, L):
        structure_factor += (
            (nk.operator.spin.sigmaz(hi, i) * nk.operator.spin.sigmaz(hi, j))
            * ((-1) ** (i - j))
            / L
        )


# %% [markdown]
# Once again, notice that we had to multiply the exchange operator (matrix) by some factor. This is to account for the Marshall basis rotation we made in our model.

# %% [markdown]
# We can now optimize our variational ansatz. The optimization data for each iteration will be stored in a log file, which contains the following information:
# 1. Mean, variance and uncertainty in the Energy $ \langle \hat{H} \rangle $
# 2. Mean, variance and uncertainty in the Energy Variance, $ \langle\hat{H}^{2}\rangle-\langle \hat{H}\rangle^{2}$.
# 3. Acceptance rates of the sampler
# 4. Mean, variance and uncertainty of observables (if specified)
#
# Now let's learn the ground-state!

# %%
# Run the optimization protocol
gs.run(out="test", n_iter=600, obs={"Structure Factor": structure_factor})

# %% [markdown]
# ## 5) Data Visualisation
#
# Now that we have optimized our machine to find the ground state of the $J_1-J_2$ model, let's look at what we have.
# The relevant data are stored in the ".log" file while the optimized parameters are in the ".wf" file. The files are all in json format.
#
# We shall extract the energy as well as specified observables (antiferromagnetic structure factor in our case) from the ".log" file.

# %%
# Load the data from the .log file
import json

data = json.load(open("test.log"))

iters = data["Energy"]["iters"]
energy = data["Energy"]["Mean"]["real"]
sf = data["Structure Factor"]["Mean"]["real"]

# %%

# %%
fig, ax1 = plt.subplots()
ax1.plot(iters, energy, color="blue", label="Energy")
ax1.set_ylabel("Energy")
ax1.set_xlabel("Iteration")
ax2 = ax1.twinx()
ax2.plot(iters, np.array(sf), color="green", label="Structure Factor")
ax2.set_ylabel("Structure Factor")
ax1.legend(loc=2)
ax2.legend(loc=1)
plt.show()

# %% [markdown]
# Let's also compute the average of those quantities (energy and neel order) over the last 50 iterations where the optimization seems to have converged.

# %%
print(
    rf"Structure factor = {np.mean(sf[-50:]):.3f}({np.std(np.array(sf[-50:]))/np.sqrt(50):.3f})"
)
print(
    rf"Energy = {np.mean(energy[-50:]):.3f}({np.std(energy[-50:])/(np.sqrt(50)):.3f})"
)

# %% [markdown]
# ## 6) Sanity Check: Exact Diagonalisation
#
# Now that we have obtained some results using VMC, it is a good time to check the quality of our results (at least for small system sizes). For this purpose, Netket provides exact diagonalisation tools.

# %%
E_gs, ket_gs = nk.exact.lanczos_ed(op, compute_eigenvectors=True)
structure_factor_gs = (
    ket_gs.T.conj() @ structure_factor.to_linear_operator() @ ket_gs
).real[0, 0]

# %% [markdown]
# Here we have specified that we want the corresponding eigenvector (in order to compute observables).

# %%
print(f"Exact Ground-state Structure Factor: {structure_factor_gs:.3f}")
print(f"Exact ground state energy = {E_gs[0]:.3f}")

# %% [markdown]
# So we see that the both energy and the structure factor we obtained is in agreement with the value obtained via exact diagonalisation.
