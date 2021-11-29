
.. _custom-models:

**********************
Defining Custom Models
**********************

In this section we give some examples on how to define models for NetKet 3.
There are mainly 3 ways to do that, and they all involve using some third party
framework.
Whatever you pick, we strongly advise you to read their documentation and whatch
some examples.

The 3 frameworks that are supported are:

* `Flax Linen API <https://flax.readthedocs.io/en/latest/examples.html>`_, which is an easy-to-use framework to define complex neural networks
* `Barebone `stax` <https://jax.readthedocs.io/en/latest/jax.example_libraries.stax.html>`_, which is an example module included with JAX.
* `Haiku <https://github.com/deepmind/dm-haiku>`_, which is a competitor to Flax, and offers somewhat equivalent expressivity with a rather different syntax.


Commonalities
-------------

Whatever the framework you pick, your model must be able to accept batches of states, so 2-Dimensonal matrices :code:`(B,N)` where :math:`N` is the number of local degrees of freedom in the hilbert space (spatial sites) and :math:`B` is the number of batches.
The result *must* be a :code:`(B,)` vector  where every element is the evaluation of
your network for that entry.

If you have a model that is differnt to write in such a way to act on batches, you
can use `jax.vmap <https://jax.readthedocs.io/en/latest/jax.html#jax.vmap>`_ to vectorize it.

Your model will be compilde with `jax.jit`. Therefore in general you should NEVER (unless you know what you are doing) use :code:`numpy`, but rather :code:`jax.numpy` inside of it.
If you want to understand why, read `Jax 101 guide <https://jax.readthedocs.io/en/latest/jax-101/index.html>`_ ( however, even if you don't care, we think it's hard to us a tool you don't undrstand: so at least rad `Jax for the Impatient <https://flax.readthedocs.io/en/latest/notebooks/jax_for_the_impatient.html>`_, which is shorter).


Defining models: init and apply functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Internally, variational states don't need a Flax model to work with, but only two functions: an initialization
function, used to initialize the parameters and the state of the model, and an apply function, used to evaluate
the model.

If you don't want to use Flax, Haiku or other supported methods, you can define your own tuple of functions and
pass it to the Variational State constructor. Keep in mind, however, that those two functions will be executed
inside :code:`jax.jit` blocks, so they must be jit-compatible.



Using Flax Linen
----------------

To define a model using Flax Linen you need to define a Flax Module. Normally those functionalities are present
in the :code:`flax.linen` module, that people usually import with :code:`import flax.linen as nn` (some day in
a few months from now, :code:`import flax.nn` will work, but at the moment it won't, as it's importing a different,
legacy, deprecated module).

Flax supports complex numbers but does not make it overly easy to work with them.
As such, netket exports a module, `netket.nn` which re-exports the functionality in `flax.nn`, but
with the additional support of complex numbers.

To define a Flax Module, simply create a class that inherits from `nn.Module`.
This class cannot have an :code:`__init__` method, but can have several class attributes.
Class attributes should be hashable objects (so in general they can be strings, numbers, other classes, but cannot
be numpy or jax arrays).

Models should define the :code:`__call__(self, x)` function that represents their action on a batch of inputs :code:`x`.

.. code:: python

	import netket.nn as nknn
	import jax.numpy as jnp

	class Model1(nknn.Module):

		y : float = 1.0

		def __call__(self, x):
			return self.y * jnp.sum(x, axis=-1)


The example above does a very simple sum on the input and multiplies it by a number. To create the module, we simply construct it
passing any optional class attribute, such as:

.. code:: python

	model = Model1(y=0.5)

If you want to use some layers inside your model, you can for example create them in the `__call__` function by decorating it with
the :code:`@nn.compact` decorator. Don't worry: they will only be initialised once.

.. code:: python

	import netket.nn as nknn
	import jax.numpy as jnp

	class RBM(nknn.Module):

		y : float = 1.0
		alpha : int = 1

		@nknn.compact
		def __call__(self, x):
			# create a dense layers with alpha * N features, where N is the size of the system
			dense = nknn.Dense(features=self.alpha*x.shape[-1])
			# apply the dense layer
			x = dense(x)
			# sum the output
			return self.y * jnp.sum(x, axis=-1)

For more advanced examples, you can check the `source-code <https://github.com/netket/netket/tree/master/netket/models>`_
of the models included in netket  or Flax documentation.

Using Jax/Stax
---------------

See tutorial :doc:`Using Jax: Netket 3 preview <../tutorials/jax>`


Using Haiku
---------------

See `this example <https://github.com/netket/netket/blob/master/Examples/Ising1d/ising1d_hk.py>`_
