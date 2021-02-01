########################
Getting Started
########################

.. currentmodule:: netket

Welcome to the documentation for Netket 3.0 (alpha).

Please read the release notes to see what has changed since the last release.

Installation and requirements
-----------------------------

Netket v3.0 requires `python>= 3.8` and optionally a recent MPI install.
To install, run one of the two following commands

.. code-block:: 

   pip install git+https://github.com/netket/netket@nk3
   pip install git+https://github.com/netket/netket@nk3#egg=netket[mpi]

The latter enables MPI-related functionalities.
Additionally, if you don't have it installed (yet) you must install `libjax`
with one of the following commands.

.. code-block:: 

	pip install -U libjax
	pip install -U jax jaxlib==0.1.59+cuda110 -f https://storage.googleapis.com/jax-releases/jax_releases.html

The latter enables nvidia-gpu support.

To query the installed `netkte` version you can run the following command in your shell

.. code-block:: 

   python -e "import netket; print(netket.version)"


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
--------------

Netket v3 API is centered around `flax <https://flax.readthedocs.io>`_, a jax library to simplify the definition and
usage of Neural-Networks.

Flax supports complex numbers but does not make it overly easy to work with them.
As such, netket exports a module, `netket.nn` which re-exports the functionality in `flax.nn`, but 
with the additional support of complex numbers.
Also `netket.optim` is a re-export of `flax.optim` with few added functionalities.

Lastly, in `netket.jax` there are a few functions, notably `jax.grad` and `jax.vjp` adapted to work with
arbitrary real or complex functions, and/or with MPI. 


Graphs
------

The documentation is organised by module:

Graphs: :ref:`graph`



poi

important:: 
   Its a note! in markdown!
   ant then this

poi  le api

API
---
The api can be [accessed here](api)


