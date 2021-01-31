Requirements
############

NetKet is a light-weight framework with minimal dependencies on external libraries.
Dependencies are discussed below, together with common strategies to install them on your machine.
In a nutshell, the only strict requirements are a working `MPI` C++ compiler, `CMake`, and a modern
Python interpreter.

MPI and CMake
"""""""""""""

In order to install NetKet you need to have a working C++11 compiler installed on your computer.
NetKet relies on `MPI <https://en.wikipedia.org/wiki/Message_Passing_Interface>`_ to provide seamless parallelism on multiples computing cores.

Below you can find more detailed, platform-dependent steps to install those requirements.

Mac
---

On Mac Os, one of the simplest strategy to get `MPI` and `CMake` is to first get `https://brew.sh <https://brew.sh>`_ and then either do:

.. code-block:: 

	brew install cmake open-mpi


to get `Open MPI <https://www.open-mpi.org>`_, or if you prefer `MPICH <https://www.mpich.org>`_ :

.. code-block:: 

	brew install cmake mpich


Ubuntu
------

On Ubuntu you can get `Open MPI <https://www.open-mpi.org>`_ and the needed development headers doing:

.. code-block:: 

	sudo apt-get install cmake libopenmpi-dev openmpi-bin openmpi-doc

Alternatively, you can have `MPICH <https://www.mpich.org>`_:

.. code-block:: 

	sudo apt-get install cmake libmpich-dev mpich


Other platforms
---------------

On other platforms/Linux distributions, it is fairly easy to find pre-compiled packages, for example you can have a look at these installation guidelines: `CMake <https://cmake.org/download/>`_, `MPICH <http://www.mpich.org/downloads/>`_.


Optional Python Libraries
"""""""""""""""""""""""""

It is also suggested to have `IPython <https://ipython.readthedocs.io/en/stable/>`_, and `matplotlib <https://matplotlib.org/>`_ installed, to fully enjoy our Tutorials and Examples.

.. code-block:: 

	pip install matplotlib jupyter

