import os
import platform
import multiprocessing
import re
import shlex
import subprocess
import sys

from distutils import log
from setuptools import setup

setup(
    name="netket",
    version="3.0",
    author="Giuseppe Carleo et al.",
    url="http://github.com/netket/netket",
    author_email="netket@netket.org",
    license="Apache 2.0",
    packages=[
        "netket",
        "netket.graph",
        "netket.hilbert",
        "netket.logging",
        "netket.machine",
        "netket.machine.density_matrix",
        "netket.sampler",
        "netket.sampler.jax",
        "netket.stats",
        "netket.operator",
        "netket.optimizer",
    ],
    long_description="""NetKet is an open - source project delivering cutting - edge
         methods for the study of many - body quantum systems with artificial
         neural networks and machine learning techniques.""",
    zip_safe=False,
    install_requires=[
        "numpy>=1.16",
        "cmake>=3.12",
        "scipy>=1.2.1",
        "mpi4py>=3.0.1",
        "tqdm>=4.42.1",
        "numba>=0.48.0",
        "networkx>=2.4",
    ],
    python_requires=">=3.6",
    extras_require={"dev": ["pytest", "python-igraph", "pre-commit", "black"],},
)
