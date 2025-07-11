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
#     display_name: mynetket
#     language: python
#     name: python3
# ---

# %% [markdown] id="oNeFQPUVCAj8"
# # Vision Transformer wave function
#
# Authors: Riccardo Rende and Luciano Loris Viteritti, February 2025
#
# The transformer architecture has become the state-of-art model for natural language processing tasks
# and, more recently, also for computer vision tasks, thus defining the Vision Transformer (ViT) architecture.
# The key feature is the ability to describe long-range correlations among the elements of the input
# sequences, through the so-called self-attention mechanism. In this tutorial, we will present the ViT wave function, an adaptation of the ViT
# architecture to define a class of variational Neural-Network Quantum States (NQS) for quantum spin
# systems (see Ref. [VT23, VIT23]).
#
#
# We begin by importing the necessary libraries, using [flax](https://github.com/google/flax)'s legacy linen interface for building neural networks.

# %% colab={"base_uri": "https://localhost:8080/"} id="cLdx0VxER_-o" outputId="2944888a-8adc-4a5d-cfac-5746ad0bb945"
import matplotlib.pyplot as plt

import netket as nk

import jax
import jax.numpy as jnp

print(jax.devices())

import flax
from flax import linen as nn

from einops import rearrange

seed = 0
key = jax.random.key(seed)


# %% [markdown] id="DkFLKj68R3_p"
# ## ViT architecture
#
# The process of constructing the amplitude corresponding to a physical spin configuration $\boldsymbol{\sigma}$ involves the following steps (see Ref. [VIT23]):
#
# 1. *Embedding*
#   + The input spin configuration $\boldsymbol{\sigma}$ is initially divided into $n$ patches. The specific shape of the patches depends on the structure of the lattice and its dimensionality, see for example Refs. [VT23, VIT23, RAS24]
#   + The patches are linearly projected into a $d$-dimensional embedding space, resulting in a sequence of vectors $(\mathbf{x}_1, \cdots, \mathbf{x}_n)$, where $\mathbf{x}_i \in \mathbb{R}^d$.
# 2. *Transformer Encoder*
#   + A Transformer Encoder with real-valued parameters processes these embedded patches, producing another sequence of vectors $(\mathbf{y}_1, \cdots, \mathbf{y}_n)$, where $\mathbf{y}_i \in \mathbb{R}^d$.
# 3. *Output layer*
#   + The hidden representation $\boldsymbol{z}$ of the configuration $\boldsymbol{\sigma}$ is defined by summing all these output vectors: $\boldsymbol{z}=\sum_{i=1}^n \mathbf{y}_i \in \mathbb{R}^d$.
#   + A fully-connected layer with complex-valued parameters maps $\boldsymbol{z}$ to a single complex number, defining the amplitude $\text{Log}[\Psi_{\theta}(\boldsymbol{\sigma})]$ corresponding to the input configuration $\boldsymbol{\sigma}$.
#
# A schematic illustration of the ViT architecture is provided in the following:
#
# ![ViTarchitecture](https://s3.gifyu.com/images/bSz0i.gif)
#
#
# In the first part of this notebook, we implement the Transformer architecture by hand to get through to the smallest details. To be concrete, we consider a spin-$1/2$ system on a two dimensional $L\times L$ square lattice.

# %% [markdown] id="I68_598ARre-"
# ## 1. Embedding
#
# We begin by writing a flax module that, given a batch of $M$ spin configurations, first splits each configuration of shape $L\times L$ into $L^2/b^2$ patches of size $b\times b$. Then, each patch is embedded in $\mathbb{R}^d$, with $d$ the *embedding dimension*.
#


# %% id="3OeYHGI-Hx27"
def extract_patches2d(x, patch_size):
    batch = x.shape[0]
    n_patches = int((x.shape[1] // patch_size**2) ** 0.5)
    x = x.reshape(batch, n_patches, patch_size, n_patches, patch_size)
    x = x.transpose(0, 1, 3, 2, 4)
    x = x.reshape(batch, n_patches, n_patches, -1)
    x = x.reshape(batch, n_patches * n_patches, -1)
    return x


class Embed(nn.Module):
    d_model: int  # dimensionality of the embedding space
    patch_size: int  # linear patch size
    param_dtype = jnp.float64

    def setup(self):
        self.embed = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )

    def __call__(self, x):
        x = extract_patches2d(x, self.patch_size)
        x = self.embed(x)

        return x


# %% colab={"base_uri": "https://localhost:8080/"} id="d8qi6voW3Eff" outputId="63b24a33-4796-4d8d-ba4c-58666b342d96"
# test embedding module implementation
d_model = 32  # embedding dimension
patch_size = 2  # linear patch size

# initialize a batch of spin configurations, considering a system on a 10x10 square lattice
M = 200
L = 10

key, subkey = jax.random.split(key)
spin_configs = jax.random.randint(subkey, shape=(M, L * L), minval=0, maxval=1) * 2 - 1

print(f"{spin_configs.shape = }")

# initialize the embedding module
embed_module = Embed(d_model, patch_size)

key, subkey = jax.random.split(key)
params_embed = embed_module.init(subkey, spin_configs)

# apply the embedding module to the spin configurations
embedded_configs = embed_module.apply(params_embed, spin_configs)

print(f"{embedded_configs.shape = }")


# %% [markdown] id="1uzzNlT15dAu"
# Working with configurations of shape $10\times 10$ and choosing a patch size of $2\times 2$, the embedding module maps each configuration into a sequence of vectors $(\mathbf{x}_1, \cdots, \mathbf{x}_n)$, with $\mathbf{x}_i \in \mathbb{R}^d$ for all $i$. In the considered setup, the resulting number of vectors is $n=25$ and we have chosen an embedding dimension of $d=32$.
#
# :::{warning}
# The function that extracts the patches from the spin configuration must be adapted to the specific lattice geometry. In particular, the function `extract_patches2d` is designed for square lattice without basis.
# :::

# %% [markdown] id="1EeJK9A85Ngr"
# ## 2. Transformer Encoder
#
# The Transformer Encoder block is composed of four ingredients: Multi-Head Attention, two-layer feed-forward neural network, layer normalization and skip connections. These elements are applied sequentially in the Encoder block as represented in the following figure:
#
# ![](https://i.ibb.co/V02p9Gst/transformer-encoder.jpg)
#
# In the following we analyze the different building blocks.
#
# ### Multi-Head Attention
#
# The core element of the Transformer architecture is the so-called *attention layer*, which processes the sequence of input vectors $(\mathbf{x}_1, \cdots, \mathbf{x}_n)$, with $\mathbf{x}_i \in \mathbb{R}^d$ for all $i$, producing a new sequence $(\boldsymbol{A}_1, \dots, \boldsymbol{A}_n)$, with $\boldsymbol{A}_i \in \mathbb{R}^d$. The goal of this transformation is to construct context-aware output vectors by combining all input vectors (see Ref. [VA17]):
#
# \begin{equation}
#     \boldsymbol{A}_i = \sum_{j=1}^n \alpha_{ij}(\boldsymbol{x}_i, \boldsymbol{x}_j) V \boldsymbol{x}_j \ .
# \end{equation}
#
# The attention weights $\alpha_{ij}(\boldsymbol{x}_i, \boldsymbol{x}_j)$ form a $n\times n$ matrix, where $n$ is the number of patches, which measure the relative importance of the $j$-$th$ input when computing the new representation of the $i$-$th$ input. To parametrize the ViT wave function, we consider a simplified attention mechanism, called *factored attention* (see Ref. [RM24, SV22]), taking the attention weights only depending on positions $i$ and $j$, but not on the actual values of the spins in these patches. In equations, factored attention leads to $\alpha_{ij}(\boldsymbol{x}_i, \boldsymbol{x}_j)=\alpha_{ij}$. Below, we show how to implement the factored attention module in flax.


# %% id="MZoldlup3rns"
class FactoredAttention(nn.Module):
    n_patches: int  # lenght of the input sequence
    d_model: int  # dimensionality of the embedding space (d in the equations)

    def setup(self):
        self.alpha = self.param(
            "alpha", nn.initializers.xavier_uniform(), (self.n_patches, self.n_patches)
        )
        self.V = self.param(
            "V", nn.initializers.xavier_uniform(), (self.d_model, self.d_model)
        )

    def __call__(self, x):
        y = jnp.einsum("i j, a b, M j b-> M i a", self.alpha, self.V, x)
        return y


# %% [markdown] id="kXuTJnqAGEvo"
# For the specific application of approximating ground states of
# quantum many-body spin Hamiltonians, factored attention yields equivalent performance with respect to the standard attention mechanism,
# while reducing the computational cost and parameter usage (see Ref.[RA25]).
#
# To improve the expressivity of the self-attention mechanism, Multi-Head attention can be considered, where for each position $i$ different attention representations $\boldsymbol{A}_i^{\mu}$ are computed, where $\mu = 1, \dots, h$ with $h$ the total number of heads. The different vectors $\boldsymbol{A}_i^{\mu} \in \mathbf{R}^{d/h}$ are computed in
# parallel, concatenated together, and linearly combined through a matrix of weights $W$.
#
# Below we build a flax module that implements the Factored Multi-Head Attention mechanism. In addition, we also provide a translational invariant implementation.
#
# :::{note}
# For approximating ground states of translationally invariant Hamiltonians, it is useful to implement translationally invariant attention mechanisms where $\alpha_{ij} = \alpha_{i-j}$.
# :::

# %% id="2EQnUHRARScb"
from functools import partial


@partial(jax.vmap, in_axes=(None, 0, None), out_axes=1)
@partial(jax.vmap, in_axes=(None, None, 0), out_axes=1)
def roll2d(spins, i, j):
    side = int(spins.shape[-1] ** 0.5)
    spins = spins.reshape(spins.shape[0], side, side)
    spins = jnp.roll(jnp.roll(spins, i, axis=-2), j, axis=-1)
    return spins.reshape(spins.shape[0], -1)


class FMHA(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.v = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        self.W = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),
            param_dtype=self.param_dtype,
        )
        if self.transl_invariant:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches),
                self.param_dtype,
            )
            sq_n_patches = int(self.n_patches**0.5)
            assert sq_n_patches * sq_n_patches == self.n_patches
            self.alpha = roll2d(
                self.alpha, jnp.arange(sq_n_patches), jnp.arange(sq_n_patches)
            )
            self.alpha = self.alpha.reshape(self.n_heads, -1, self.n_patches)
        else:
            self.alpha = self.param(
                "alpha",
                nn.initializers.xavier_uniform(),
                (self.n_heads, self.n_patches, self.n_patches),
                self.param_dtype,
            )

    def __call__(self, x):
        # apply the value matrix in paralell for each head
        v = self.v(x)

        # split the representations of the different heads
        v = rearrange(
            v,
            "batch n_patches (n_heads d_eff) -> batch n_patches n_heads d_eff",
            n_heads=self.n_heads,
        )

        # factored attention mechanism
        v = rearrange(
            v, "batch n_patches n_heads d_eff -> batch n_heads n_patches d_eff"
        )
        x = jnp.matmul(self.alpha, v)
        x = rearrange(
            x, "batch n_heads n_patches d_eff  -> batch n_patches n_heads d_eff"
        )

        # concatenate the different heads
        x = rearrange(
            x, "batch n_patches n_heads d_eff ->  batch n_patches (n_heads d_eff)"
        )

        # the representations of the different heads are combined together
        x = self.W(x)

        return x


# %% colab={"base_uri": "https://localhost:8080/"} id="LM8DtDRQRSQD" outputId="c837d2dd-6e2b-4385-88d6-f4e479036663"
# test Factored MultiHead Attention module
n_heads = 8  # number of heads
n_patches = embedded_configs.shape[1]  # lenght of the input sequence

# initialize the Factored Multi-Head Attention module
fmha_module = FMHA(d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_fmha = fmha_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
attention_vectors = fmha_module.apply(params_fmha, embedded_configs)

print(f"{attention_vectors.shape = }")


# %% [markdown] id="pCeUca5HX-c3"
# ### Encoder Block
#
# In each encoder block, the MultiHead attention mechanism is followed by a two-layers feed-forward neural network. Layer normalization and skip connections are also added to stabilize the training of deep architectures.


# %% id="6N23TCYHYLH7"
class EncoderBlock(nn.Module):
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False
    param_dtype = jnp.float64

    def setup(self):
        self.attn = FMHA(
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=self.n_patches,
            transl_invariant=self.transl_invariant,
        )

        self.layer_norm_1 = nn.LayerNorm(param_dtype=self.param_dtype)
        self.layer_norm_2 = nn.LayerNorm(param_dtype=self.param_dtype)

        self.ff = nn.Sequential(
            [
                nn.Dense(
                    4 * self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
                nn.gelu,
                nn.Dense(
                    self.d_model,
                    kernel_init=nn.initializers.xavier_uniform(),
                    param_dtype=self.param_dtype,
                ),
            ]
        )

    def __call__(self, x):
        x = x + self.attn(self.layer_norm_1(x))

        x = x + self.ff(self.layer_norm_2(x))
        return x


# %% [markdown] id="8BjwIlRLbzQS"
# Based on this block, we can implement a module for the full Transformer Encoder applying a sequence of encoder blocks. The number of these blocks is defined by the number of layers of the Transformer architecture.


# %% id="bHWROcz5bud5"
class Encoder(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    n_patches: int  # lenght of the input sequence
    transl_invariant: bool = False

    def setup(self):
        self.layers = [
            EncoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                n_patches=self.n_patches,
                transl_invariant=self.transl_invariant,
            )
            for _ in range(self.num_layers)
        ]

    def __call__(self, x):

        for l in self.layers:
            x = l(x)

        return x


# %% colab={"base_uri": "https://localhost:8080/"} id="uGTDosLzcGkk" outputId="82172451-fa4f-4ea1-9980-77227ff9dfab"
# test Transformer Encoder module
num_layers = 4  # number of layers

# initialize the Factored Multi-Head Attention module
encoder_module = Encoder(num_layers, d_model, n_heads, n_patches)

key, subkey = jax.random.split(key)
params_encoder = encoder_module.init(subkey, embedded_configs)

# apply the Factored Multi-Head Attention module to the embedding vectors
x = embedded_configs
y = encoder_module.apply(params_encoder, x)

print(f"{y.shape = }")

# %% [markdown] id="ed8aW_HrcilR"
# The Transformer Encoder processes the embedded patches $(\mathbf{x}_1, \cdots, \mathbf{x}_n)$, with $\mathbf{x}_i \in \mathbb{R}^d$, producing another sequence of vectors $(\mathbf{y}_1, \cdots, \mathbf{y}_n)$, with $\mathbf{y}_i \in \mathbb{R}^d$.

# %% [markdown] id="U_8fwA1XdaPy"
# ## 3. Output layer
# For each configuration $\boldsymbol{\sigma}$, we compute its hidden representation $\boldsymbol{z}=\sum_{i=1}^n \mathbf{y}_i$. Then, we produce a single complex number representing its amplitude using the fully-connected layer defined in Ref. [CS17]:
#
# \begin{equation}
#     \text{Log}[\Psi(\boldsymbol{\sigma})] = \sum_{\alpha=1}^d \log\cosh \left( b_{\alpha} + \boldsymbol{w}_{\alpha} \cdot \boldsymbol{z} \right) \ ,
# \end{equation}
#
# :::{note}
# The parameters $\{ b_\alpha, \boldsymbol{w}_\alpha \}$ are taken to be complex valued, contrary to the parameters of the Transformer Encoder which are all real valued.
# :::

# %% id="GVuWv-BRfcnT"
log_cosh = (
    nk.nn.activation.log_cosh
)  # Logarithm of the hyperbolic cosine, implemented in a more stable way


class OuputHead(nn.Module):
    d_model: int  # dimensionality of the embedding space
    param_dtype = jnp.float64

    def setup(self):
        self.out_layer_norm = nn.LayerNorm(param_dtype=self.param_dtype)

        self.norm2 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )
        self.norm3 = nn.LayerNorm(
            use_scale=True, use_bias=True, param_dtype=self.param_dtype
        )

        self.output_layer0 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )
        self.output_layer1 = nn.Dense(
            self.d_model,
            param_dtype=self.param_dtype,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=jax.nn.initializers.zeros,
        )

    def __call__(self, x):

        z = self.out_layer_norm(x.sum(axis=1))

        out_real = self.norm2(self.output_layer0(z))
        out_imag = self.norm3(self.output_layer1(z))

        out = out_real + 1j * out_imag

        return jnp.sum(log_cosh(out), axis=-1)


# %% [markdown] id="ajN6baa2gn_b"
# Combining the Embedding, Encoder and OutputHead modules we can implement a module for the full ViT architecture.


# %% id="leWFMMrNhF-e"
class ViT(nn.Module):
    num_layers: int  # number of layers
    d_model: int  # dimensionality of the embedding space
    n_heads: int  # number of heads
    patch_size: int  # linear patch size
    transl_invariant: bool = False

    @nn.compact
    def __call__(self, spins):
        x = jnp.atleast_2d(spins)

        Ns = x.shape[-1]  # number of sites
        n_patches = Ns // self.patch_size**2  # lenght of the input sequence

        x = Embed(d_model=self.d_model, patch_size=self.patch_size)(x)

        y = Encoder(
            num_layers=self.num_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_patches=n_patches,
            transl_invariant=self.transl_invariant,
        )(x)

        log_psi = OuputHead(d_model=self.d_model)(y)

        return log_psi


# %% colab={"base_uri": "https://localhost:8080/"} id="dYar0GnThjMH" outputId="6a62bea6-845e-4f12-b5a6-70b83237ac37"
# test ViT module

# initialize the ViT module
vit_module = ViT(num_layers, d_model, n_heads, patch_size)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

# apply the ViT module
log_psi = vit_module.apply(params, spin_configs)

print(f"{log_psi.shape = }")

# %% [markdown] id="F4fLHmeZ4cDN"
# ## Ground state optimization
#
# We show how to train the ViT wave function on the two-dimensional $J_1$- $J_2$ Heisenberg model on a $10\times 10$ square lattice.
# The system is described by the following Hamiltonian (with periodic boundary conditions):
#
# $$
#     \hat{H} = J_1 \sum_{\langle {\boldsymbol{r}},{\boldsymbol{r'}} \rangle} \hat{\boldsymbol{S}}_{\boldsymbol{r}}\cdot\hat{\boldsymbol{S}}_{\boldsymbol{r'}}
#     + J_2 \sum_{\langle \langle {\boldsymbol{r}},{\boldsymbol{r'}} \rangle \rangle} \hat{\boldsymbol{S}}_{\boldsymbol{r}}\cdot\hat{\boldsymbol{S}}_{\boldsymbol{r'}} \ .
# $$
#
# We fix $J_2/J_1=0.5$ and we use the VMC_SRt driver (see Ref.[RAS24]) implemented in NetKet to optimize the ViT wave function.

# %% colab={"base_uri": "https://localhost:8080/"} id="sP1jBIXb6H_W" outputId="1b432288-79e0-4caf-cd84-00b16d2d3dae"
seed = 0
key = jax.random.key(seed)

L = 10
n_dim = 2
J2 = 0.5

lattice = nk.graph.Hypercube(length=L, n_dim=n_dim, pbc=True, max_neighbor_order=2)

# Hilbert space of spins on the graph
hilbert = nk.hilbert.Spin(s=1 / 2, N=lattice.n_nodes, total_sz=0)

# Heisenberg J1-J2 spin hamiltonian
hamiltonian = nk.operator.Heisenberg(
    hilbert=hilbert, graph=lattice, J=[1.0, J2], sign_rule=[False, False]
).to_jax_operator()  # No Marshall sign rule

# Intiialize the ViT variational wave function
vit_module = ViT(
    num_layers=4, d_model=60, n_heads=10, patch_size=2, transl_invariant=True
)

key, subkey = jax.random.split(key)
params = vit_module.init(subkey, spin_configs)

# Metropolis Local Sampling
N_samples = 4096
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert,
    graph=lattice,
    d_max=2,
    n_chains=N_samples,
    sweep_size=lattice.n_nodes,
)

optimizer = nk.optimizer.Sgd(learning_rate=0.0075)

key, subkey = jax.random.split(key, 2)
vstate = nk.vqs.MCState(
    sampler=sampler,
    model=vit_module,
    sampler_seed=subkey,
    n_samples=N_samples,
    n_discard_per_chain=0,
    variables=params,
    chunk_size=512,
)

N_params = nk.jax.tree_size(vstate.parameters)
print("Number of parameters = ", N_params, flush=True)

# Variational monte carlo driver
from netket.experimental.driver.vmc_srt import VMC_SRt

vmc = VMC_SRt(
    hamiltonian=hamiltonian,
    optimizer=optimizer,
    diag_shift=1e-4,
    variational_state=vstate,
    jacobian_mode="complex",
)


# %% colab={"base_uri": "https://localhost:8080/", "height": 86, "referenced_widgets": ["a8dbb3f874ac4f8baac39b8ebe18fd9d", "25f3f464d5a24e9284ba5ea140f4944a", "3ceb6d5780a24fd2bb8f203bca36b661", "feb52ce513814bbd8ada812695fc1718", "d72add58f069432a898851ed1ad825da", "2345d611347c4c239a92a245589ac1a8", "550f50e1dd7f4cdc8131ee99638cfdc7", "f3970deece314655b106677a604e82ad", "175b3b3073bb46a0b90867e1b833e49f", "fb9ff265fcbd48bbad4931702d59eeb1", "9d68301885f0446db22eb115d8ff948b"]} id="NaSPVYboM3K-" outputId="4f534963-f32f-4a99-f2dd-76e1e7180940"
# Optimization
log = nk.logging.RuntimeLog()

N_opt = 800
vmc.run(n_iter=N_opt, out=log)

# %% [markdown] id="gcZOaRr3VIj7"
# :::{note}
# The previous cell requires approximately two hours to run on a single A100 GPU.
# :::
#
# We can visualize the training dynamics by plotting the mean energy as a function of the optimization steps:

# %% colab={"base_uri": "https://localhost:8080/", "height": 449} id="pJLZjm6JR65_" outputId="749046b9-38e1-4eba-d614-3137664c225c"
energy_per_site = log.data["Energy"]["Mean"].real / (L * L * 4)

print("Last value: ", energy_per_site[-1])

plt.plot(energy_per_site)

plt.xlabel("Iterations")
plt.ylabel("Energy per site")

plt.show()

# %% [markdown] id="apySasQ1V9mC"
# The final variational energy obtained is approximately $E_{\text{var}} \approx -0.4964$. To improve this result, longer simulations with a larger number of samples should be conducted on several GPUs in parallel.

# %% [markdown] id="efM4dYrc2aei"
# ## Pretrained model (Hugging Face)
#
# We provide a pretrained Vision Transformer (ViT) architecture with $P = 434760$ parameters at [Hugging Face ViT](https://huggingface.co/nqs-models/j1j2_square_10x10_05). Below, we demonstrate how the model can be easily downloaded and used within the NetKet framework with just a few lines of code.

# %% colab={"base_uri": "https://localhost:8080/", "height": 595, "referenced_widgets": ["1b191cfa8c23430cb5fdf2c3bc7c4f3f", "4a1dfd6b5dc24ea18d764336ea22c73b", "d12a4d807c084a2e8ada609222451593", "35b177119bd649ff9c2eb4e7b7d16006", "05f3fe3cf76a4b208cb61cd313a81d84", "d3bd0f3007a74db49aa9871f27318403", "c0547a61bab744dc8bdd1961392db1a4", "198659bb5be343388bb96525a1d2a716", "039a69c2c5dc45cdb41f20a370fe691a", "6c85dddaa458418a95e0a0a0e8a16848", "a0b8d94847fc412394d600a71806fbd3", "26299012fe9948fd84e494ab2cf6a5f9", "f7eceef080aa49b393658306e7b1eeb6", "540be49a24a642d58cc194a4a7513ede", "4d3b112f42534b79be249a793a7a218a", "d1e381dcef4448608c3a9ff9528994af", "a2b99659c19049c3a20236ee35f2bcc9", "a378ba5272c04401b53dffdd2d8ad4ed", "7a871b6a985d40979a7c3fa862284c3c", "d7225a56d9bd46bcb5d7c1cc098ac43a", "870b858a65fb4ffcab5640381daa4843", "2aa8f77212b84a559b437840904d7404", "369a12d13e914409869048258a20d3a5", "60b798acf66b4a7596c9b071cac1560e", "be49f75ea338407bbadbf5cd38c9f48d", "b932fa821a8544a4a5b6ff0504914ca9", "491cfb2653e84bc9a37ce01099350a85", "0e66e84b9a964a5ca0816b0b2c14e20b", "46680451d39b4a17969585065685d5f0", "cedf9000b7284e64a15936e64f836764", "14f5b7cef4534f6d9e53d9cd29882120", "aa99850e2b814fd2ae0e60be6bc0e384", "5c5db6dc04df4d6db8a81514e637f1e2", "12d075e39a9e40b7996c0038161b4169", "5cc80eb516e44849957adf7eaaed0239", "61481d82173a474f98b609be520c5885", "e4187851ae4b4a988f51609941f5772c", "6ef382e6ac034d39bdbfa83e0344d05f", "0bfb2379fed34779807500f0a82bd88c", "8d4c4de19ead40f883d97a51861175ec", "e62420499402411ea6146890ab429fbc", "aa812a9cd8a746308b035bf82bf0c16e", "bde53ef374ce4f74a745f69af6154eca", "8ba2d8d18ce1423cb7b76a39b6ad3546", "f5b90c96d94f4f2ea7f0015645f9ed52", "6d150846c07f4768898a94cf91c6f290", "1fd2890752c04801ac1c18dfa4025b5b", "8ecbfd6b13bf4a898aabf511b4780e24", "3b2cc4650c44418fbae7f5ef21b83131", "1ef554a7cd984085b4b8b14a5cb4cae3", "258cac78c8f6409ba176ecb8c6675c0a", "5452539416fc4f8ea529860712440410", "e38d9e14fbd64f958553a5477ff2ffbd", "0ae26697bc034569b4c613c5121c5dc9", "8d0dd88721734f17aea4d33409885afb", "ea8439c781b54b0ebf9f8667529b9554", "00340b865499426a8b6bbb56de8062d2", "5046f05716484443b3164fe8a75cf6eb", "b4b0bdcbf31c4b8c8a45eeb0bfe59dfb", "861140b2c5ff4ba7a3108345e18d3707", "05395b4807fd4d51b258e062d0ee75de", "9141e150a5fa44d198eabe35fe6782f3", "b3712136160f4413a7e23a6b155adf04", "4a71faafccc4455aa13790872e421a12", "d18b3e9f5e0e43e4a288d43726d43288", "8f437dc404604157a0a6716089cb6c14"]} id="luWZoc3ze3U7" outputId="fe354171-a2b4-41eb-d8fc-5009f4cdd5f0"
from transformers import FlaxAutoModel

wf = FlaxAutoModel.from_pretrained(
    "nqs-models/j1j2_square_10x10", trust_remote_code=True
)
N_params = nk.jax.tree_size(wf.params)
print("Number of parameters = ", N_params, flush=True)

# %% colab={"base_uri": "https://localhost:8080/"} id="_vajEkBMe9zu" outputId="d31f93fd-bbcd-466a-b146-85d6fbea41f7"
sampler = nk.sampler.MetropolisExchange(
    hilbert=hilbert, graph=lattice, d_max=2, n_chains=16384, sweep_size=lattice.n_nodes
)

vstate = nk.vqs.MCState(
    sampler=sampler,
    apply_fun=wf.__call__,
    sampler_seed=subkey,
    n_samples=16384,
    n_discard_per_chain=0,
    variables=wf.params,
    chunk_size=16384,
)

# Overwrite samples with already thermalized ones
from huggingface_hub import hf_hub_download
from flax.training import checkpoints

flax.config.update("flax_use_orbax_checkpointing", False)

path = hf_hub_download(repo_id="nqs-models/j1j2_square_10x10", filename="spins")
samples = checkpoints.restore_checkpoint(ckpt_dir=path, prefix="spins", target=None)
samples = jnp.array(
    samples, dtype="int8"
)  #! some versions of netket do not require this line

vstate.sampler_state = vstate.sampler_state.replace(σ=samples)

E = vstate.expect(hamiltonian)

print(E)

# %% [markdown] id="KtYQifV3goHY"
# Using a single A100 GPU, the previous cell should run in less than one minute and produce a mean variational energy of approximately $E_{\text{var}} \approx -0.497508$. This energy can be further improved by enforcing lattice symmetries in the variational wavefunction, reaching $E_{\text{var}} \approx -0.497676335$ when restoring translational, point group, and parity symmetries (see [Hugging Face ViT](https://huggingface.co/nqs-models/j1j2_square_10x10_05) for more details).

# %% [markdown] id="bh5-oPREBh_8"
# ---
#
# ### References
#
# + [CS17] Carleo, Giuseppe, and Matthias Troyer. "Solving the quantum many-body problem with artificial neural networks." Science 355, no. 6325 (2017): 602-606.
# + [VT23] Viteritti, Luciano Loris, Riccardo Rende, and Federico Becca. "Transformer variational wave functions for frustrated quantum spin systems." Physical Review Letters 130, no. 23 (2023): 236401.
# + [VIT23] Viteritti, Luciano Loris, Riccardo Rende, Alberto Parola, Sebastian Goldt, and Federico Becca. "Transformer wave function for the shastry-sutherland model: emergence of a spin-liquid phase." arXiv preprint arXiv:2311.16889 (2023).
# + [VA17] Vaswani, A. "Attention is all you need." Advances in Neural Information Processing Systems (2017).
# + [RM24] Rende, Riccardo, Federica Gerace, Alessandro Laio, and Sebastian Goldt. "Mapping of attention mechanisms to a generalized Potts model." Physical Review Research 6, no. 2 (2024): 023057.
# + [SV22] S. Jelassi, M. E. Sander, and Y. Li, "Vision transformers provably learn spatial structure", in Advances in neural information processing systems (2022)
# + [RA25] Rende, Riccardo, and Luciano Loris Viteritti. "Are queries and keys always relevant? A case study on transformer wave functions." Machine Learning: Science and Technology 6, no. 1 (2025): 010501.
# + [RAS24] Rende, Riccardo, Luciano Loris Viteritti, Lorenzo Bardone, Federico Becca, and Sebastian Goldt. "A simple linear algebra identity to optimize large-scale neural network quantum states." Communications Physics 7, no. 1 (2024): 260.

# %% [markdown]
#
