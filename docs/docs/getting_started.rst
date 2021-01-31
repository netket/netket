########################
Getting Started
########################

.. currentmodule:: netket

Welcome to the documentation for Netket 3.0....

Please read the release notes to see what has changed since the last release.

To query the installed version you can run the following command in your shell

.. code-block:: 

   python -e "import python; python.version()"


Introduction 
------------

Netket is a numerical framework written in Python to simulate many-body quantum systems using
variational methods. In general, netket allows the user to parametrize quantum states using 
arbitrary functions, be it simple mean-field ansatze, Jastrow, MPS ansatze or convolutional
neural networks.
Those states can be sampled efficiently in order to estimate observables or other quantities.
Stochastic optimisation of the energy or a time-evolution are implemneted on top of those samplers.

.. math::
   a + b \frac{x+1}{\int_x^y}

.. math:: 
   a + b \frac{x+1}{\int_x^y}

Installation
------------

Netket can be installed by using pip or anaconda. 
We reccomend to use pip 

graph
.. toctree::
   docs/graphs
   docs/api

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


