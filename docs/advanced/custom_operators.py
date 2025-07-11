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
#     display_name: python-3.11.2
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Defining Custom Observables and Operators
#
# In this page we will show how to define custom operators and the relevant methods used to compute expectation values and their gradients.
# This page assumes you have already read the [Hilbert module](Hilbert) and [Operator module](Operator) documentation and have a decent understanding of their class hierarchy.

# %% tags=["hide-cell"]
# %pip install netket --quiet

# %% [markdown]
# If you want to define new quantities to be computed over variational states, such as the infidelity, a Renyi entropy or very peculiar observables, you can always write a function yourself that uses the attributes of variational states such as `vstate._apply_fun`, which is the function used to evaluate the log-wavefunction, `vstate.samples`, which returns a set of (possibly cached) samples and `vstate.variables` which returns the variational parameters.
#
# For example, when working with a standard {class}`~netket.vqs.MCState` you might try the following:

# %%
import jax.numpy as jnp
import netket as nk


def expect_avg_X(vstate):
    """Compute average magnetization along X axis."""
    # this only works with spins
    assert isinstance(vstate.hilbert, nk.hilbert.Spin)

    samples = vstate.samples

    samples_flipped = -samples

    # compute the local observable
    Oloc = jnp.exp(vstate.log_val(samples_flipped) - vstate.log_val(samples))

    # compute the expectation value
    return nk.stats.statistics(Oloc)


# %% [markdown]
# However this approach is not ideal. This will not work with existing drivers such as {class}`nk.driver.VMC`, and it will not be consistent with the existing interface. Moreover, if you also want it to work with the Full-Summation variational state {class}`nk.vqs.FullSumState` you will have to implement another function and use it accordingly.
#
# For this reason, NetKet exposes an extensible interface to support arbitrary Observables and Operators within its {meth}`~nk.vqs.VariationalState.expect` and {meth}`~nk.vqs.VariationalState.expect_and_grad` methods.
#
# The most general interface is to define a custom {class}`netket.experimental.observable.AbstractObservable` and its expectation and gradient dispatch rules. This gives you total control and is least-likely to cause bugs.
# However, if your observable is an operator over a discrete hilbert space and can easily be converted to a density matrix and its expectation value can be written easily, you can also use more advanced base classes such as {class}`netket.operator.DiscreteOperator`.

# %% [markdown]
# ## Defining a custom _zero_ operator
#
# Let's assume you want to define an operator that always returns 0.
# That's a very useful operator!
#

# %%
from netket.experimental.observable import AbstractObservable


class ZeroOperator(AbstractObservable):
    @property
    def dtype(self):
        return float


# %% [markdown]
# To define an operator we always need to define the `dtype` property, representing the dtype of the output.
#
# Moreover, since we did not define the `__init__()` method, we inherit the `AbstractObservable` init method which
# requires an hilbert space to be specified.
# First we define the hilbert space, then we construct the operator itself.

# %%
hi = nk.hilbert.Spin(0.5, 4)
hi

# %%
zero_op = ZeroOperator(hi)
zero_op

# %% [markdown]
# If we define a variational state on the same hilbert space, and we try to compute the expectation value on that space, an error will be thrown:

# %%
vs = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM())

# vs.expect(zero_op)

# %% [markdown]
# The error should look roughly like the following:
# ```python
# >>> vs.expect(zero_op)
#
# NotFoundLookupError: get_local_kernel_arguments(MCState(
#   hilbert = Spin(s=1/2, N=4),
#   sampler = MetropolisSampler(rule = LocalRule(), n_chains = 16, sweep_size = 4, reset_chains = False, machine_power = 2, dtype = <class 'float'>),
#   n_samples = 1008,
#   n_discard_per_chain = 100,
#   sampler_state = MetropolisSamplerState(rng state=[3154305721 3544831998]),
#   n_parameters = 24), ZeroOperator(hilbert=Spin(s=1/2, N=4), dtype=<class 'float'>)) could not be resolved.
#
# Closest candidates are:
# get_local_kernel_arguments(vstate: netket.vqs.MCState, Ô: netket.operator._lazy.Squared)
#     <function get_local_kernel_arguments at 0x2830c3a60> @ ~/Dropbox/Ricerca/Codes/Python/netket/netket/vqs/mc/mc_state/expect.py:43
# get_local_kernel_arguments(vstate: netket.vqs.MCState, Ô: netket.operator.DiscreteOperator)
#     <function get_local_kernel_arguments at 0x28310fb00> @ ~/Dropbox/Ricerca/Codes/Python/netket/netket/vqs/mc/mc_state/expect.py:57
# get_local_kernel_arguments(vstate: netket.vqs.MCState, Ô: netket.operator.DiscreteJaxOperator)
#     <function get_local_kernel_arguments at 0x28310fc40> @ ~/Dropbox/Ricerca/Codes/Python/netket/netket/vqs/mc/mc_state/expect.py:71
# ```

# %% [markdown]
# This is because you defined a new operator, but you did not define how to compute an expectation value with it.
# The method to use to compute an expectation value is chosen from a list using multiple-dispatch, and it is defined both by the type of the variational state and by the type of the operator.
#
# To define the expect method you should do the following:
#

# %%
import numpy as np


@nk.vqs.expect.dispatch
def expect_zero_operator(vstate: nk.vqs.MCState, op: ZeroOperator, chunk_size: None):
    return np.array(0, dtype=op.dtype)


# %% [markdown]
# Dispatch is based on [plum-dispatch's multiple-dispatch](https://beartype.github.io/plum/intro.html) logic which is inspired by Julia's function definition.
# The underlying idea is that we often want to define functions that work for specific combination of types, such in this case.
#
# The three arguments to expect must closely match the convention used by NetKet:
#  - The first argument should be the variational state, and the type annotation `: nk.vqs.MCState` must specify the class of the variational state for which your algorithm works. In this case, we are working with `MCState` so that is the one we will use;
#  - The second argument must be the operator or observable to be computed, and the type annotation must specify what class your implementation will work with. In this case we use `: ZeroOperator` because we are implementing a function that computes the expectation value of a `ZeroOperator` over a `MCState`;
#  - The third argument must be the specified `chunk_size` by which we compute the values. This is either a `None`, if no chunking is supported, an integer, if only chunking is supported or `Optional[int]` for optional chunking. In this case we specify `None` because we don't support chunking.

# %%
vs.expect(zero_op)

# %% [markdown]
# Now that you defined the dispatch rule for computing the expectation value of this observable, you might also want to define the dispatch rule for the gradient, which is normally accessed by calling `variational_state.expect_and_grad(operator)`.
# To define it, you can follow the same procedure as above:

# %%
import jax


@nk.vqs.expect_and_grad.dispatch
def expect_and_grad_zero_operator(
    vstate: nk.vqs.MCState, op: ZeroOperator, chunk_size: None, **kwargs
):
    # this is the expectation value, as before
    expval = np.array(0, dtype=op.dtype)
    # this is the gradient, which of course is zero.
    grad = jax.tree.map(jnp.zeros_like, vstate.parameters)
    return expval, grad


# %% [markdown]
# The syntax to define the `expect_and_grad` rule is the same as before, but you should also accept arbitrary keyword arguments. Those may contain additional options specified by the variational state (such as mutable arguments) that you can use or ignore in your calculation.

# %%
vs.expect_and_grad(zero_op)


# %% [markdown]
# ## Defining an operator from scratch
#
# Let's try to reimplement an operator from scratch.
# I will take the $\hat{X} = \sum_i^N \hat{\sigma}^{(X)}_i $ operator as a simple example.
#
# In Variational Monte Carlo, we usually compute expectation values through the following expectation value:
#
# $$ \langle \hat{X} \rangle = \langle \psi | \hat{X} | \psi \rangle = \sum_\sigma |\psi(\sigma)|^2 \sum_{\eta} \frac{\langle\sigma|\hat{X}|\eta\rangle\langle\eta|\psi\rangle}{\langle \sigma | \psi\rangle} = \mathbb{E}_{\sigma \approx |\psi(\sigma)|^2}\left[ E^{loc}(\sigma)\right]
# $$
#
# where $ E^{loc}(\sigma) =  \sum_{\eta} \frac{\langle\sigma|\hat{X}|\eta\rangle\langle\eta|\psi\rangle}{\langle \sigma | \psi\rangle} $ is called the local estimator.
#
# First we define the operator itself:
#


# %%
class XOperator(AbstractObservable):
    @property
    def dtype(self):
        return float


# %% [markdown]
# To then compute expectation values we need the following methods:
#
#  - A method to sample the probability distribution $|\psi(\sigma)|^2$. This is already provided by monte carlo variational state interface and samples can be retrieved simply by calling {attr}`~netket.vqs.MCState.samples`.
#  - A method to take the samples $ \sigma $ and compute the connected elements $ \eta $ so that $ \langle\sigma|\hat{X}|\eta\rangle \neq 0 $. This should also return those matrix elements.
#  - A method to compute the local energy given the matrix elements, the $\sigma$ and $ \eta $ and the variational state.
#  - The statistical average of the local energies.
#
# First we implement a method returning all the connected elements.
# Given a bitstring $ \sigma $ for $N$ spins, the connected elements are $N$ bitstrings of the same length, where each one has a flipped bit.
# The matrix element is always 1
#

# %%
from functools import partial  # partial(sum, axis=1)(x) == sum(x, axis=1)


@jax.vmap
def get_conns_and_mels(sigma):
    # this code only works if sigma is a single bitstring
    assert sigma.ndim == 1

    # get number of spins
    N = sigma.shape[-1]
    # repeat eta N times
    eta = jnp.tile(sigma, (N, 1))
    # diagonal indices
    ids = np.diag_indices(N)
    # flip those indices
    eta = eta.at[ids].set(-eta.at[ids].get())
    return eta, jnp.ones(N)


@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def e_loc(logpsi, pars, sigma, eta, mels):
    return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)


# %% [markdown]
# The first function takes a single bitstring $ \sigma $, a vector with $N$ entries, and returns a batch of bitstrings $ eta_i $, a matrix $ N \times N $ where every element in the diagonal is flipped.
# We then used `jax.vmap` to make this function work with batches of inputs $\sigma$ (we could have written it from the beginning to work on batches, but this way the meaning is very clear).
#
# The other function also takes a single $ \sigma $ and a batch of $ \eta $ and their matrix elements, and uses the formula above to compute the local energy.
# This function is also `jax.vmap`ped in order to work with batches of inputs. The argument `in_axes=(None, None, 0,0,0)` means that the first 2 arguments do not change among batches, while the other three are batched along the first dimension.
#
# With those two functions written, we can write the expect method
#


# %%
@nk.vqs.expect.dispatch
def expect(vstate: nk.vqs.MCState, op: XOperator, chunk_size: None):
    return _expect(vstate._apply_fun, vstate.variables, vstate.samples)


@partial(jax.jit, static_argnums=0)
def _expect(logpsi, variables, sigma):
    n_chains = sigma.shape[-2]
    N = sigma.shape[-1]
    # flatten all batches
    sigma = sigma.reshape(-1, N)

    eta, mels = get_conns_and_mels(sigma)

    E_loc = e_loc(logpsi, variables, sigma, eta, mels)

    # reshape back into chains to compute statistical information
    E_loc = E_loc.reshape(-1, n_chains)

    # this function computes things like variance and convergence information.
    return nk.stats.statistics(E_loc)


# %% [markdown]
# The dispatch rule is a thin layer that calls a jitted function, in order to have more speed.
# We cannot directly jit `expect` itself because it takes as input a `vstate`, which is not directly
# jit-compatible.
#
#
# The internal, jitted `_expect` takes as input the `vstate._apply_fun` function which is the one evaluating the neural quantum state, together with it's inputs (`vstate.variables`).
# Note that if you want to make the `expect_and_grad` method work with this custom operator, you will also have to define the dispatch rule for `expect_and_grad`.
#
# We can now test this operator:

# %%
x_op = XOperator(hi)
display(x_op)

vs.expect(x_op)

# %% [markdown]
# We can compare it with the built-in LocalOperator implementation in NetKet whose indexing is written in Numba. Of course, as the operators are identical, the expectation values will match.

# %%
x_localop = sum([nk.operator.spin.sigmax(hi, i) for i in range(hi.size)])
display(x_localop)

vs.expect(x_localop)

# %% [markdown]
# It might be interesting to investigate the performance difference among the two operators:

# %%
# %timeit vs.expect(x_op)
# %timeit vs.expect(x_localop)

# %% [markdown]
# And you can see that our jax-based approach is more efficient than NetKet's built-in operators.
# This is for two reasons: first, NetKet's LocalOperator is a general object that can handle completely arbitrary operators, and this flexibility comes at the price of a slightly lower performance.
# Usually this is not an issue because if your model is complication enough, the cost of indexing the operator will be negligible with respect to the cost of evaluating the model itself.
#
# However, writing the operator in Jax has the added benefit that XLA (the Jax compiler) can introspect the code of the operator, and might uber-optimise the evaluation of the expectation value.

# %% [markdown]
# ## Defining an operator the easy way
#
# Most operator that can be used efficiently with VMC approaches have the same structure as above, therefore, when defining new operators you will often find yourself redefining a similar function, where the only thing changing is the code to compute the connected elements, matrix elements, and the local kernel (in the case above computing E_loc).
# If you also want to define the gradient, the boilerplate becomes considerable.
#
# As such, to reduce the amount of boilerplate code that users must write when defining custom operators, NetKet will attempt by default to use a default expectation kernel.
# This kernel looks very similar to the `_expect` function above, and will call a function similar to `get_conns_and_mels` and a local kernel, selected using multiple dispatch.
#
# In the example below we show how to make use of this _lean_ interface.
#

# %%
from netket.operator import AbstractOperator


class XOperatorLean(AbstractOperator):
    @property
    def dtype(self):
        return float

    @property
    def is_hermitian(self):
        return True


def e_loc(logpsi, pars, sigma, extra_args):
    eta, mels = extra_args
    # check that sigma has been reshaped to 2D, eta is 3D
    # sigma is (Nsamples, Nsites)
    assert sigma.ndim == 2
    # eta is (Nsamples, Nconnected, Nsites)
    assert eta.ndim == 3

    # let's write the local energy assuming a single sample, and vmap it
    @partial(jax.vmap, in_axes=(0, 0, 0))
    def _loc_vals(sigma, eta, mels):
        return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, sigma)), axis=-1)

    return _loc_vals(sigma, eta, mels)


@nk.vqs.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: XOperatorLean):
    return e_loc


@nk.vqs.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: XOperatorLean):
    sigma = vstate.samples
    # get the connected elements. Reshape the samples because that code only works
    # if the input is a 2D matrix
    extra_args = get_conns_and_mels(sigma.reshape(-1, vstate.hilbert.size))
    return sigma, extra_args


# %% [markdown]
# The first function here is very similar to the `e_loc` we defined in the previous section, however instead of taking `eta` and `mels` as two different arguments, they are passed as a tuple named `extra_args`, and must be unpacked by the kernel.
# That's because every operator has it's own `get_local_kernel_arguments` which prepares those extra_args, and every different operator might be passing different objects to the kernel.
#
# Also do note that the kernel is jit-compiled, therefore it must make use of `jax.numpy` functions, while the `get_local_kernel_argument` function *is not jitted*, and it is executed before jitting.
# This is in order to allow extra flexibility: sometimes your operators need some pre-processing code that is not jax-jittable, but is only numba-jittable, for example (this is the case of most operators in NetKet).
#
# All operators and super-operators in netket define those two methods.
# If you want to see some examples of how it is used internally, look at the source code found in [this folder](https://github.com/netket/netket/blob/master/netket/vqs/mc/MCState/expect.py).
# An additional benefit of using this latter definition, is that is automatically enables `expect_and_grad` for your custom operator.
#
# We can now test this new implementation:

# %%
x_op_lean = XOperatorLean(hi)
vs.expect(x_op_lean)

# %% [markdown]
# ## Comparison of the two approaches
#
# Above you've seen two different ways to define `expect`, but it might be unclear which one you should use in your code.
# In general, you should prefer the second one: the _simple_ interface is easier to use and enables by default gradients too.
#
# If you can express the expectation value of your operator in such a way that it can be estimated by simply averaging a single local-estimator the simple interface is best.
# We expect that you should be able to use this interface in the vast majority of cases.
# However in some cases you might want to try something particular, experiment, or your operator cannot be expressed in this form, or you want to try out some optimisations (such as chunking), or compute many operators at once.
# That's why we also allow you to override the general `expect` method alltogether: it allows a lot of extra flexibility.
