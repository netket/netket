(custom-operator)=
# Defining Custom Operators

In this page we will show how to define custom operators and the relevant methods used to compute expectation values and their gradients.
This page assumes you have already read the [Hilbert module](Hilbert) and [Operator module](Operator) documentation and have a decent understanding of their class hierarchy.


(defining-custom-zero)=
## Defining a custom _zero_ operator

Let's assume you want to define an operator that always returns 0.
That's a very useful operator!

```python
import netket as nk
from netket.operator import AbstractOperator

class ZeroOperator(AbstractOperator):
	
	@property
	def dtype(self):
		return float

```

To define an operator we always need to define the `dtype` property, representing the dtype of the output. 
Moreover, since we did not define the `__init__()` method, we inherit the `AbstractOperator` init method which
requires an hilbert space to be specified.

```python
>>> hi = nk.hilbert.Spin(0.5, 4)
Spin(s=1/2, N=2)

>>> zero_op = ZeroOperator(hi)
ZeroOperator(hilbert=Spin(s=1/2, N=2))
```

If we define a variational state on the same hilbert space, and we try to compute the expectation value on that space, an error will be thrown:

```python
>>> vs  = nk.vqs.MCState(nk.sampler.MetropolisLocal(hi), nk.models.RBM())
MCState(
  hilbert = Spin(s=1/2, N=2),
  sampler = MetropolisSampler(rule = LocalRule(), n_chains = 16, machine_power = 2, n_sweeps = 2, dtype = <class 'numpy.float64'>),
  n_samples = 1008,
  n_discard_per_chain = 100,
  sampler_state = MetropolisSamplerState(rng state=[3684014404 2614644650]),
  n_parameters = 8)

>>> vs.expect(zero_op)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/filippovicentini/Dropbox/Ricerca/Codes/Python/netket/netket/vqs/base.py", line 164, in expect
    return expect(self, Ô)
  File "plum/function.py", line 537, in plum.function.Function.__call__
  File "/home/filippovicentini/Dropbox/Ricerca/Codes/Python/netket/netket/vqs/mc/mc_state/expect.py", line 96, in expect
    σ, args = get_local_kernel_arguments(vstate, Ô)
  File "plum/function.py", line 536, in plum.function.Function.__call__
  File "plum/function.py", line 501, in plum.function.Function.resolve_method
  File "plum/function.py", line 435, in plum.function.Function.resolve_signature
plum.function.NotFoundLookupError: For function "get_local_kernel_arguments", signature Signature(netket.vqs.mc.mc_state.state.MCState, __main__.ZeroOperator) could not be resolved.
```

This is because you defined a new operator, but you did not define how to compute an expectation value with it.
The method to use to compute an expectation value is chosen from a list using multiple-dispatch, and it is defined both by the type of the variational state and by the type of the operator.

To define the expect method you should do the following:
```python
>>> @nk.vqs.expect.dispatch
>>> def expect(vstate: nk.vqs.MCState, op: ZeroOperator):
>>> 	return np.array(0, dtype=op.dtype)
```

And if we now call again the expect method of the variational state, it will work!
```python
>>> vs.expect(zero_op)
array(0.)
````

(defining-custom-discrete-operator)=
## Defining an operator from scratch

Let's try to reimplement an operator from scratch. 
I will take the $\hat{X} = \sum_i^N \hat{\sigma}^{(X)}_i $ operator as a simple exapmle.

In Variational Monte Carlo, we usually compute expectation values through the following expectation value:

$$

\langle \hat{X} \rangle = \langle \psi | \hat{X} | \psi \rangle = \sum_\sigma |\psi(\sigma)|^2 \sum_{\eta} \frac{\langle\sigma|\hat{X}|\eta\rangle\langle\eta|\psi\rangle}{\langle \sigma | \psi\rangle} = \mathbb{E}_{\sigma \approx |\psi(\sigma)|^2}\left[ E^{loc}(\sigma)\right]

$$

where $ E^{loc}(\sigma) =  \sum_{\eta} \frac{\langle\sigma|\hat{X}|\eta\rangle\langle\eta|\psi\rangle}{\langle \sigma | \psi\rangle} $ is called the local estimator.

First we define the operator itself:

```python
class XOperator(AbstractOperator):
  @property
  def dtype(self):
    return float

```

To then compute expectation values we need the following methods:
 
 - A method to sample the probability distribution $|\psi(\sigma)|^2$. This is already provided by monte carlo variational state interface and samples can be retrieved simply by calling {attr}`~netket.vqs.MCState.samples`.
 - A method to take the samples $ \sigma $ and compute the connected elements $ \eta $ so that $ \langle\sigma|\hat{X}|\eta\rangle \neq 0 $. This should also return those matrix elements.
 - A method to compute the local energy given the matrix elements, the $\sigma$ and $ \eta $ and the variational state.
 - The statistical average of the local energies.

First we implement a method returning all the connected elements.
Given a bitstring $ \sigma $ for $N$ spins, the connected elements are $N$ bistrings of the same length, where each one has a flipped bit.
The matrix element is always 1

```python
@jax.vmap
def get_conns_and_mels(sigma):
  # get number of spins
  N = sigma.shape[-1]
  # repeat eta N times
  eta = jnp.tile(sigma, (N,1))
  # diagonal indices
  ids = np.diag_indices(N)
  # flip those indices
  eta = eta.at[ids].set(-eta.at[ids].get())
  return eta, jnp.ones(1)

@partial(jax.vmap, in_axes=(None, None, 0,0,0))
def e_loc(logpsi, pars, sigma, eta, mels):
  return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, eta)), axis=-1)
``` 

The first function takes a single bitstring $ \sigma $, a vector with $N$ entries, and returns a batch of bitstrings $ eta_i $, a matrix $ N\cross N $ where every element in the diagonal is flipped.
We then used `jax.vmap` to make this function work with batches of inputs $\sigma$ (we could have written it from the beginning to work on batches, but this way the meaning is very clear).

The other function also takes a single $ \sigma $ and a batch of $ \eta $ and their matrix elements, and uses the formula above to compute the local energy. 
This function is also `jax.vmap`ped in order to work with batches of inputs. The argument `in_axes=(None, None, 0,0,0)` means that the first 2 arguments do not change among batches, while the other three are batched along the first dimension.

With those two functions written, we can write the expect method

```python
@nk.vqs.expect.dispatch
def expect(vstate: nk.vqs.MCState, op: XOperator):
  return _expect(vstate._apply_fun, vstate.variables, vstate.samples)

@partial(jax.jit, static_argnums=0)
def _expect(logpsi, variables, sigma):
  n_chains = sigma.shape[-2]
  N = sigma.shape[-1]
  # flatten all batches
  sigma = sigma.reshape(-1, N)

  eta, mels = get_conns_and_mels(sigma)

  E_loc = e_loc(logpsi, variables, sigma, eta, mels)

  # reshape back into chains to compute statistical informations
  E_loc = E_loc.reshape(-1, n_chains)

  # this function computes things like variance and convergence informations.
  return nk.stats.statistics(E_loc.T)
```

The dispatch rule is a thin layer that calls a jitted function, in order to have more speed.
We cannot directly jit `expect` itself because it takes as input a `vstate`, which is not directly
jit-compatible.


The internal, jitted `_expect` takes as input the `vstate._apply_fun` function which is the one evaluating
the neural quantum state, together with it's inputs (`vstate.variables`).

Note that if you want to make the `expect_and_grad` method work with this custom operator, you will also have to define the dispatch rule for `expect_and_grad`.

(defining-custom-discrete-operator-easy)=
## Defining an operator the easy way

Most operator that can be used efficiently with VMC approaches have the same structure as above, therefore, when defining new operators you will often find yourself redefining a similar function, where the only thing changing is the code to compute the connected elements, matrix elements, and the local kernel (in the case above computing E_loc).
If you also want to define the gradient, the boilerplate becomes considerable.

As such, to reduce the amount of boilerplate code that users must write when defining custom operators, NetKet will attempt by default to use a default expectation kernel.
This kernel looks very similar to the `_expect` function above, and will call a function similar to `get_conns_and_mels` and a local kernel, selected using multiple dispatch.

In the example below we show how to make use of this _lean_ interface.

```python
class XOperatorLean(AbstractOperator):
  @property
  def dtype(self):
    return float

@partial(jax.vmap, in_axes=(None, None, 0,0,0))
def e_loc(logpsi, pars, sigma, extra_args):
  eta, mels = extra_args
  return jnp.sum(mels * jnp.exp(logpsi(pars, eta) - logpsi(pars, eta)), axis=-1)

@nk.vqs._mc.get_local_kernel.dispatch
def get_local_kernel(vstate: nk.vqs.MCState, op: XOperatorLean):
  return e_loc

@nk.vqs._mc.get_local_kernel_arguments.dispatch
def get_local_kernel_arguments(vstate: nk.vqs.MCState, op: XOperatorLean):
  sigma = vstate.sigma
  extra_args = get_conns_and_mels(sigma)
  return sigma, extra_args
```

The first function here is very similar to the `e_loc` we defined in the previous section, however instead of taking `eta` and `mels` as two different arguments, they are passed as a tuple named `extra_args`, and must be unpacked by the kernel.
That's because every operator has it's own `get_local_kernel_arguments` which prepares those extra_args, and every different operator might be passing different objects to the kernel.

Also do note that the kernel is jit-compiled, therefore it must make use of `jax.numpy` functions, while the `get_local_kernel_argument` function *is not jitted*, and it is executed before jitting.
This is in order to allow extra flexibility: sometimes your operators need some pre-processing code that is not jax-jittable, but is only numba-jittable, for example (this is the case of most operators in NetKet). 

All operators and super-operators in netket define those two methods. 
If you want to see some examples of how it is used internally, look at the source code found in [this folder](https://github.com/netket/netket/blob/master/netket/vqs/mc/MCState/expect.py).
An additional benefit of using this latter definition, is that is automatically enables `expect_and_grad` for your custom operator.

## Comparison of the two approaches

Above you've seen two different ways to define `expect`, but it might be unclear which one you should use in your code.
In general, you should prefer the second one: the _simple_ interface is easier to use and enables by default gradients too.

If you can express the expectation value of your operator in such a way that it can be estimated by simply averaging a single local-estimator the simple interface is best. 
We expect that you should be able to use this interface in the vast majority of cases. 
However in some cases you might want to try something particular, experiment, or your operator cannot be expressed in this form, or you want to try out some optimisations (such as chunking), or compute many operators at once.
That's why we also allow you to override the general `expect` method alltogether: it allows a lot of extra flexibility.