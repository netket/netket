from setuptools import setup, find_packages

DEV_DEPENDENCIES = [
    "pytest>=6",
    "pytest-xdist>=2",
    "coverage>=5",
    "pytest-cov>=2.10.1",
    "networkx~=2.4",
    "flaky>=3.7",
    "pre-commit",
    "black==21.6b0",
    "flakehell>=0.9",
]
MPI_DEPENDENCIES = ["mpi4py>=3.0.1, <4", "mpi4jax~=0.3.1"]
TENSORBOARD_DEPENDENCIES = ["tensorboardx>=2.0.0"]
BASE_DEPENDENCIES = [
    "numpy~=1.18",
    "scipy~=1.5",
    "tqdm~=4.56",
    "plum-dispatch~=1.3",
    "numba>=0.52, <0.54",
    "python-igraph~=0.9",
    "jax>0.2.16, <=0.2.17",
    "jaxlib>=0.1.69",
    "flax>=0.3.0, <0.4",
    "orjson~=3.4",
    "optax>=0.0.2, <0.0.10",
]

setup(
    name="netket",
    author="Giuseppe Carleo et al.",
    url="http://github.com/netket/netket",
    author_email="netket@netket.org",
    license="Apache 2.0",
    summmary="Netket : Machine Learning techniques for many-body quantum systems.",
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
    packages=find_packages(),
    install_requires=BASE_DEPENDENCIES,
    python_requires=">=3.7",
    extras_require={
        "dev": DEV_DEPENDENCIES,
        "mpi": MPI_DEPENDENCIES,
        "tensorboard": TENSORBOARD_DEPENDENCIES,
        "all": MPI_DEPENDENCIES + DEV_DEPENDENCIES + TENSORBOARD_DEPENDENCIES,
    },
)
