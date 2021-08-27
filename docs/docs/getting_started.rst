***************
Getting Started
***************

.. currentmodule:: netket

Welcome to the documentation for Netket 3.0.

Please read the release notes to see what has changed since the last release.

Installation and requirements
-----------------------------

Netket v3.0 requires `python>= 3.8` and optionally a recent MPI install.
To install, run one of the two following commands

.. code:: bash 

   pip install netket

If you want to run NetKet on a GPU, you must install a GPU-compatible :code:`jaxlib`. For that, we advise you to
look at the instructions on `jax repository <https://github.com/google/jax#pip-installation>`_, however at the time
of writing, this means you should run the following command: 

.. code:: bash 

    pip install --upgrade "jax[cuda111]" -f https://storage.googleapis.com/jax-releases/jax_releases.html

Where the jaxlib version must correspond to the version of the existing CUDA installation you want to use. Refer to jax
documentation to learn more about matching cuda versions with python wheels.
   
To query the installed `netket` version you can run the following command in your shell

.. code:: bash 

   python -e "import netket; print(netket.version)"


MPI
***

If you want to use MPI, you will need to have a working MPI compiler. You can install the
dependencies necessary to run with MPI with the following command:

.. code:: bash

   pip install netket[mpi]

Subsequently, NetKet will exploit MPI-level parallelism for the Monte-Carlo sampling.
See :ref:`this block <warn-mpi-sampling>` to understand how NetKet behaves under MPI.

Introduction 
------------

Netket is a numerical framework written in Python to simulate many-body quantum systems using
variational methods. In general, netket allows the user to parametrize quantum states using 
arbitrary functions, be it simple mean-field ansatze, Jastrow, MPS ansatze or convolutional
neural networks.
Those states can be sampled efficiently in order to estimate observables or other quantities.
Stochastic optimisation of the energy or a time-evolution are implemnented on top of those samplers.

Netket tries to follow the `functional programming <https://en.wikipedia.org/wiki/Functional_programming>`_ paradigm, 
and is built around `jax <https://en.wikipedia.org/wiki/Functional_programming>`_. While it is possible
to run the examples without knowledge of `jax`_, we strongly reccomend getting familiar with it if you 
wish to extend netket.

This documentation is divided into several modules, each explaining in-depth how a sub-module of netket works.
You can select a module from the list on the left, or you can read the following example which contains links
to all relevant parts of the documentation.


Jax/Flax extensions
-------------------

Netket v3 API is centered around `flax <https://flax.readthedocs.io>`_, a jax library to simplify the definition and
usage of Neural-Networks. 
If you want to define more complex custom models, you should read Flax documentation on how to define a Linen module.
However, you can also use :code:`jax.stax` or `haiku <https://github.com/deepmind/dm-haiku>`_.

Flax supports complex numbers but does not make it overly easy to work with them.
As such, netket exports a module, `netket.nn` which re-exports the functionality in `flax.nn`, but 
with the additional support of complex numbers.
Also `netket.optim` is a re-export of `flax.optim` with few added functionalities.

Lastly, in `netket.jax` there are a few functions, notably `jax.grad` and `jax.vjp` adapted to work with arbitrary real or complex functions, and/or with MPI. 


Legacy API support (API before 2021)
------------------------------------

With the 3.0 official release in the beginning of 2021, we have drastically 
changed the API of Netket, which are no longer compatible with the old version.

Netket will ship a copy of the old API and functionalities under the `legacy` 
submodule. To keep using your old scripts you should change your import at the top
from `import netket as nk` to `import netket.legacy as nk`. 

While you can keep using the legacy module, we will remove it sometime soon with
version 3.1, so we strongly advise to update your scripts to the new version.
To aid you in updating your code, a lot of deprecation warning will be issued when
you use the legacy api suggesting you how to update your code.

While it might be annoying, the new API allows us to have less code to maintain
and grants more freedom to the user when defining models, so it will be a huge
improvement.

Some documentation of the legacy module can be found in this section :ref:`Legacy`, 
but please be advised that it is no longer-supported and documentation will 
probably be of poor quality.

For more information on new features and API changes, please consult :ref:`Whats New`.

.. warning:: 
    If you were using the previous version of NetKet, we strongly advise you to read
    :ref:`Whats New` as it lists several changes that might otherwise pass unnoticed.


Commented Example
-----------------

.. code-block:: python

    import netket as nk
    import numpy as np

The first thing to do is import NetKet. We usually shorten it to `nk`.

.. code-block:: python

    g = nk.graph.Hypercube(length=20, n_dim=1, pbc=True)

Then, one must define the systeme to be studied. To do so, the first
thing to do is usually defining the lattice of the model. This is not
always required, but it can sometimes avoid errors.
Seveeral types of Lattices (graphs) are defined in the :ref:`Graph` 
submodule.

In the example above we chose a 1-Dimensional chain with 20 sites and
periodic boundary conditions.

.. code-block:: python

    hi = nk.hilbert.Spin(s=1 / 2, N=g.n_nodes)
    ha = nk.operator.Ising(hilbert=hi, graph=g, h=1.0)

Then, one must define the hilbert space and the hamiltonian. Common options 
for the Hilbert spacee are :ref:`Spin`,  :ref:`Fock` or :ref:`QuBit`, but it is
also possible to define your own. Those classes are contained in the :ref:`Hilbert` 
submodule.

The hamiltonian sub-module contains several pre-built hamiltonian, such as 
:ref:`Ising` and :ref:`Bose-Hubbard`, but you can also build the operators
yourself by summing all the local terms. See the :ref:`Operators` documentation
for more informations.

.. code-block:: python

    ma = nk.models.RBM(alpha=1, dtype=float)

    sa = nk.sampler.MetropolisLocal(hi, n_chains=16)


Then, one must chose the model to use as a Neural Quantum State. Netket provides
a few pre-built models in the :ref:`Models` sub-module. 
Netket models are simply `Flax`_ modules: check out the :ref:`define-your-model` 
section for more informations on how to define or use custom models. 
We specify :code:`dtype=float` (which is the default, but we want to show
it to you) which means that weights will be stored as double-precision.
We advise you that Jax (and therefore netket) does not follow completely the standard NumPy
promotion rules, instead treating :code:`float` as a weak double-precision type
which can _loose_ precision in some cases. 
This can happen if you mix single and double precision in your models and the sampler and
is described in `Jax:Type promotion semantics <https://jax.readthedocs.io/en/latest/type_promotion.html>`_.

Hilbert space samplers are defined in the :ref:`Sampler` submodule. In general 
you must provide the constructor the hilbert space to be sampled and some options. 
In this case we ask for 16 markov chains. 
The default behaviour for samplers is to output states with double precision, but
this can be configured by specifying the :code:`dtype` argument when constructing the
sampler.
Samples don't need double precision at all, so it makes sense to use the lower 
precision, but you have to be careful with the dtype of your model in order
not to reduce the precision.

.. code-block:: python

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=0.01)


You can then chose an optimizer from the :ref:`optimizer` submodule. You can also 
use an arbitrary flax optimiser, or define your own.  

.. code-block:: python

    # Variational monte carlo driver
    gs = nk.VMC(ha, op, sa, ma, n_samples=1000, n_discard_per_chain=100)

    gs.run(n_iter=300, out=None)

Once you have all the pieces together, you can construct a variational monte
carlo optimisation driver by passing the constructor the hamiltonian and thee 
optimmmiser (which must always be the first two arguments), and theene the
sampler, mahcine and various options.

Once that is done, you can run the simulation by calling the :ref:`run` method
in the driver, specifying the output loggers and the number of iterations in
the optimisation.
