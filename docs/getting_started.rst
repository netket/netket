.. title:: Getting Started 

.. container:: jumbotron jumbotron-fluid landing-page-box landing-page-box-logo

    .. rst-class:: h1 

      **Getting Started!**

    .. rst-class:: h2 

      In few easy steps


.. container:: inner-content

    .. dropdown:: Make sure you have Python 3.7 or higher installed on Mac or Linux

        If Python 3.7 is not available on your computer, don't despair! 
        You can use `pyenv <https://github.com/pyenv/pyenv>`_ (the easiest way to install it is with the `pyenv installer <https://github.com/pyenv/pyenv-installer>`_) to install any Python version, or you can use Anaconda, even though the latter is not recomended if you plan on using MPI.

        Windows is not natively supported because Jax does not yet support it. However, if you use WSL (Windows Subsystem for Linux) NetKet will run smoothly. 

    .. dropdown:: If you plan to use MPI, make sure you have an up to date version of the `mpicc` compilers available on your path.

        When using MPI, we recommend not to use Anaconda unless it's for small experimentation on a laptop. This is due to a dependency of netket, mpi4jax. You can read more about the limitations on the `mpi4jax documentation <https://mpi4jax.readthedocs.io/en/latest/installation.html>`_. 

    .. dropdown:: :code:`pip install netket`
       Conda is also supported, but not reccomended. However you can use a conda environment and install netket with pip inside this environment.

       If you want to use MPI, use :code:`pip install netket[mpi]`.

       If you want to develop netket, extra development dependencies are installed by running :code:`pip install netket[all,dev]`

    .. dropdown:: Explore our `Tutorials <tutorials.html>`_ and check out our `Examples <https://github.com/netket/netket/tree/master/Examples>`_ 

       Tutorials are commented python notebooks. Examples are sample python files that can be run.
       If you want to experiment with mpi, try running some examples with :code:`mpirun -np2 python path/to/example.py`
