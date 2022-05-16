from setuptools import setup, find_packages

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-xdist[psutil]>=2",
    "pytest-cov>=2.10.1",
    "pytest-json-report>=1.3",
    "coverage>=5",
    "networkx~=2.4",
    "pre-commit>=2.7",
    "black==22.3.0",
    "flake8==4.0.1",
]
MPI_DEPENDENCIES = ["mpi4py>=3.0.1, <4", "mpi4jax~=0.3.1"]
EXTRA_DEPENDENCIES = ["tensorboardx>=2.0.0", "openfermion>=1.0.0"]
BASE_DEPENDENCIES = [
    "numpy~=1.18",
    "scipy>=1.5.3, <2",
    "tqdm~=4.60",
    "plum-dispatch~=1.5.1",
    "numba>=0.52, <0.57",
    "igraph~=0.9.8",
    "jax>=0.2.25, <0.4",
    "jaxlib>=0.1.72",
    "flax>=0.4, <0.5",
    "orjson~=3.4",
    "optax>=0.1.1, <0.2",
    "numba4jax>=0.0.10, <0.1",
]

setup(
    name="netket",
    author="Giuseppe Carleo et al.",
    url="http://github.com/netket/netket",
    author_email="netket@netket.org",
    license="Apache 2.0",
    description="Netket : Machine Learning techniques for many-body quantum systems.",
    long_description="""NetKet is an open-source project delivering cutting-edge
         methods for the study of many-body quantum systems with artificial
         neural networks and machine learning techniques.""",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Unix",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    packages=find_packages(include=["netket*"]),
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.7",
    extras_require={
        "dev": DEV_DEPENDENCIES,
        "mpi": MPI_DEPENDENCIES,
        "extra": EXTRA_DEPENDENCIES,
        "all": MPI_DEPENDENCIES + DEV_DEPENDENCIES + EXTRA_DEPENDENCIES,
    },
)
