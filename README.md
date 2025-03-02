<div align="center">
<img src="https://www.netket.org/logo/logo_simple.jpg" alt="logo" width="400"></img>
</div>

# __NetKet__

[![Powered by NumFOCUS](https://img.shields.io/badge/powered%20by-NumFOCUS-orange.svg?style=flat&colorA=E1523D&colorB=007D8A)](https://numfocus.org)
[![Release](https://img.shields.io/github/release/netket/netket.svg)](https://github.com/netket/netket/releases)
[![Paper (v3)](https://img.shields.io/badge/paper%20%28v3%29-arXiv%3A2112.10526-B31B1B)](https://scipost.org/SciPostPhysCodeb.7/pdf)
[![codecov](https://codecov.io/gh/netket/netket/branch/master/graph/badge.svg?token=gzcOlpO5lB)](https://codecov.io/gh/netket/netket)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg)](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)

NetKet is an open-source project delivering cutting-edge methods for the study
of many-body quantum systems with artificial neural networks and machine learning techniques.
It is a Python library built on [JAX](https://github.com/google/jax).

NetKet is an affiliated project to [numFOCUS](https://numfocus.org).

- **Homepage:** <https://www.netket.org>
- **Citing:** <https://www.netket.org/cite/>
- **Documentation:** <https://netket.readthedocs.io/en/latest/index.html>
- **Tutorials:** <https://netket.readthedocs.io/en/latest/tutorials/gs-ising.html>
- **Examples:** <https://github.com/netket/netket/tree/master/Examples>
- **Source code:** <https://github.com/netket/netket>

## Installation and Usage

NetKet runs on MacOS and Linux. We recommend to install NetKet using `pip`, but it can also be installed with `conda`.
It is often necessary to first update `pip` to a recent release (`>=20.3`) in order for upper compatibility bounds to be considered and avoid a broken installation.
For instructions on how to install the latest stable/beta release of NetKet see the [Get Started](https://www.netket.org/get_started/) page of our website or run the following command (Apple M1 users, follow that link for more instructions):

```sh
pip install --upgrade pip
pip install --upgrade netket
```

If you wish to install the current development version of NetKet, which is the master branch of this GitHub repository, together with the additional dependencies, you can run the following command:

```sh
pip install --upgrade pip
pip install 'git+https://github.com/netket/netket.git#egg=netket[all]'
```

To speed-up NetKet-computations, even on a single machine, you
can install the MPI-related dependencies by using `[mpi]` between square brackets.

```sh
pip install --upgrade pip
pip install --upgrade "netket[mpi]"
```

We recommend to install NetKet with all it's extra dependencies, which are documented below.
However, if you do not have a working MPI compiler in your PATH this installation will most likely fail because
it will attempt to install `mpi4py`, which enables MPI support in netket.

The latest release of NetKet is always available on PyPi and can be installed with `pip`.
NetKet is also available on conda-forge, however the version available through `conda install`
can be slightly out of date compared to PyPi.
To check what is the latest version released on both distributions you can inspect the badges at the top of this readme.

### Extra dependencies
When installing `netket` with pip, you can pass the following extra variants as square brakets. You can install several of them by separating them with a comma.
 - `"[dev]"`: installs development-related dependencies such as black, pytest and testing dependencies
 - `"[mpi]"`: Installs `mpi4py` to enable multi-process parallelism. Requires a working MPI compiler in your path
 - `"[extra]"`: Installs `tensorboardx` to enable logging to tensorboard, and openfermion to convert the QubitOperators.
 - `"[all]"`: Installs all extra dependencies

### MPI Support
To enable MPI support you must install [mpi4jax](https://github.com/PhilipVinc/mpi4jax). Please note that we advise to install mpi4jax  with the same tool (conda or pip) with which you install it's dependency `mpi4py`.

To check whether MPI support is enabled, check the flags
```python
>>> import netket
>>> netket.utils.mpi.available
True
```

## Getting Started

To get started with NetKet, we recommend you give a look at our [tutorials page](https://netket.readthedocs.io/en/latest/tutorials/gs-ising.html), by running them on your computer or on [Google Colaboratory](https://colab.research.google.com).
There are also many example scripts that you can download, run and edit that showcase some use-cases of NetKet, although they are not commented.

If you want to get in touch with us, feel free to open an issue or a discussion here on GitHub, or to join the MLQuantum slack group where several people involved with NetKet hang out. To join the slack channel just accept [this invitation](https://join.slack.com/t/mlquantum/shared_invite/zt-19wibmfdv-LLRI6i43wrLev6oQX0OfOw)

## License

[Apache License 2.0](https://github.com/netket/netket/blob/master/LICENSE)
