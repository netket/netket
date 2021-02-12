from setuptools import setup, find_packages

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-xdist>=2",
    "coverage>=5",
    "pytest-cov>= 2.10.1",
    "python-igraph",
    "pre-commit",
    "black==20.8b1",
]
MPI_DEPENDENCIES = ["mpi4py>=3.0.1", "mpi4jax>=0.2.9"]
BASE_DEPENDENCIES = [
    "numpy>=1.16",
    "scipy>=1.5.2",
    "tqdm>=4.56.2",
    "numba>=0.49.0",
    "networkx>=2.4",
    "jax>=0.2.9",
    "flax>=0.3.0",
]

setup(
    name="netket",
    version="3.0",
    author="Giuseppe Carleo et al.",
    url="http://github.com/netket/netket",
    author_email="netket@netket.org",
    license="Apache 2.0",
    packages=find_packages(),
    long_description="""NetKet is an open - source project delivering cutting - edge
         methods for the study of many - body quantum systems with artificial
         neural networks and machine learning techniques.""",
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.7",
    extras_require={
        "dev": DEV_DEPENDENCIES,
        "mpi": MPI_DEPENDENCIES,
        "all": MPI_DEPENDENCIES + DEV_DEPENDENCIES,
    },
)
