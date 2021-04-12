# Copyright 2021 The NetKet Authors - All rights reserved.
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

from functools import partial
from inspect import signature

import jax
import numpy as np
import jax.numpy as jnp

from netket import jax as nkjax
from netket.legacy.machine._jax_utils import outdtype
from netket.legacy.machine import Jax


# The following dicts store some 'properties' of cost functions. The keys are jitted
# cost functions. Access should be performed in jit blocks in order to be 0-cost.

# batch_axes for jax.vmap of cost functions
_batch_axes = {}

# unjitted version of the cost function
_unjitted_fun = {}

# Whever the cost function has in general complex output, or only real
_outdtype = {}


def define_local_cost_function(
    fun,
    static_argnums=0,
    batch_axes=None,
):
    """
    @define_local_cost_function(fun, static_argnums=0, batch_axes=automatic,
                                outdtype=complex)

    A decorator to be used to define a local cost function and it's gradient. The
    function to be decorated must be a jax-compatible function, that takes the following
    positional arguments:
     - The variational function evaluating a single input and returning a scalar output.
     - A pytree of parameters for the neural network function. Gradients will be
     computed with respect to this argument
     - N additional positional arguments (non static) containing any additional data.

    In order to support batching, one must also define the batch_axes variable according
    to jax vmap documentation. By default `batch_axesw=(None, None, 0...)`, meaning that
    no batching is performed for the first two arguments (the network and the
    parameters) and batching is performed along the 0-th dimension of all arguments.

    The optional kwarg `outdtype` specifies the output type (float or complex) of the
    local cost function. If you know the output is real, then `outdtype=float` will
    allow for a faster code-path.

    An example is provided below:
    ```python
    @partial(define_local_cost_function, static_argnums=0,
                                                    batch_axes=(None, None, 0, 0, 0))
    def local_energy_kernel(logpsi, pars, vp, mel, v):
        return jax.numpy.sum(mel * jax.numpy.exp(logpsi(pars, vp) - logpsi(pars, v)))
    ```
    """
    jitted_fun = jax.jit(fun, static_argnums=static_argnums)

    ig = signature(jitted_fun)
    npars = len(ig.parameters)
    if npars < 2:
        raise ValueError("Local cost functions should have at least 2 parameters.")

    # If batch_axes is not specified, assume that all parameters except the first two
    # (function and parameters) are to be batched upon.
    if batch_axes is None:
        batch_axes = (None, None) + tuple([None for _ in range(npars - 2)])

    _batch_axes[jitted_fun] = batch_axes
    _unjitted_fun[jitted_fun] = fun

    return jitted_fun


# In the code below, we define a jitted function taking as argument the
# jax_forward and pytree parametersm and a standard python function
# taking as argument the full machine. The stable API only involves the
# one taking a full JaxMachine, as it allows us to add some logic (like
# for real/complex valued machines) in the future if needed without
# breaking the API.

# In the following, all functions _jitted functions assume that the arguments
# are passed in that order:
# 0 - (static) local_cost_fun (for example local_energy kernel).
# 1 - (static) the nn function
# 2 - weights for the nn function in pytree format (directions of the gradient)
# 3 - various parameters
# Also assumes that args 1..N are the args (in that order) of local_cost_fun


@partial(jax.jit, static_argnums=(0, 1))
def _local_cost_function(local_cost_fun, logpsi, pars, *args):
    local_cost_fun_vmap = jax.vmap(
        _unjitted_fun[local_cost_fun],
        in_axes=_batch_axes[local_cost_fun],
        out_axes=0,
    )

    return local_cost_fun_vmap(logpsi, pars, *args)


def local_cost_function(local_cost_fun, model, pars, *args):
    """
    local_cost_function(local_cost_fun, machine, *args)

    Function to compute the local cost function in batches for the parameters of
    `machine`.

    Args:
        local_cost_fun: the cost function
        machine: netket's JaxMachine containing the variational ansatz and parameters
        *args: additional arguments

    Returns:
        the value of log_psi with parameters `pars` for the batches *args
    """
    return _local_cost_function(local_cost_fun, model, pars, *args)


# Starting from the 4th argument, it's the same arguments as the cost function itself
# dtype: dtype of pars
# outdtype: dtype of logpsi(pars, *args)
def __local_cost_and_grad_function(local_cost_fun, logpsi, pars, *args):
    lcfun_u = partial(_unjitted_fun[local_cost_fun], logpsi)

    der_local_cost_fun = nkjax.value_and_grad(lcfun_u, argnums=0)
    return der_local_cost_fun(pars, *args)


_local_cost_and_grad_function = jax.jit(
    __local_cost_and_grad_function, static_argnums=(0, 1)
)


def local_cost_and_grad_function(local_cost_fun, machine, parameters, *args):
    """
    local_cost_and_grad_function(local_cost_fun, machine, *args)

    Function to compute the gradient and value of the local cost function, with respect
    to the parameters of the `machine`.

    Args:
        local_cost_fun: the cost function
        machine: netket's JaxMachine containing the variational ansatz and parameters
        *args: additional arguments

    Returns:
        the value of the local_cost_fun(machine, *args)
        the gradient of of `local_cost_fun(machine, *args)`
    """
    return _local_cost_and_grad_function(local_cost_fun, machine, parameters, *args)


@partial(jax.jit, static_argnums=(0, 1))
def _local_costs_and_grads_function(local_cost_fun, logpsi, pars, *args):
    local_costs_and_grads_fun = jax.vmap(
        __local_cost_and_grad_function,
        in_axes=(None,) + _batch_axes[local_cost_fun],
        out_axes=(0, 0),
    )
    return local_costs_and_grads_fun(local_cost_fun, logpsi, pars, *args)


def local_costs_and_grads_function(local_cost_fun, machine, parameters, *args):
    """
    local_costs_and_grads_function(local_cost_fun, machine, *args)

    Function to compute the value and the gradient of the `local_cost_fun` function
    with respect to the parameters of the `machine`, vmapped along `*args` 0-th
    dimension.

    Args:
        local_cost_fun: the cost function
        machine: netket's JaxMachine containing the variational ansatz and parameters
        *args: additional arguments

    Returns:
        the value of the local_cost_function for every `*args` input
        the gradient with respect to the weights
    """
    return _local_costs_and_grads_function(
        local_cost_fun,
        machine,
        parameters,
        *args,
    )


@partial(define_local_cost_function, static_argnums=0, batch_axes=(None, None, 0, 0, 0))
def local_value_cost(logpsi, pars, vp, mel, v):
    return jnp.sum(mel * jnp.exp(logpsi(pars, vp) - logpsi(pars, v)))


@partial(define_local_cost_function, static_argnums=0, batch_axes=(None, None, 0, 0, 0))
def local_value_op_op_cost(logpsi, pars, σp, mel, σ):

    σ_σp = jax.vmap(lambda σp, σ: jnp.hstack((σp, σ)), in_axes=(0, None))(σp, σ)
    σ_σ = jnp.hstack((σ, σ))
    return jnp.sum(mel * jnp.exp(logpsi(pars, σ_σp) - logpsi(pars, σ_σ)))
