from setuptools import setup, find_packages


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
    install_requires=[
        "numpy>=1.16",
        "scipy>=1.5.2",
        "tqdm>=4.42.1",
        "numba>=0.49.0",
        "networkx>=2.4",
    ],
    python_requires=">=3.6",
    extras_require={
        "dev": ["pytest", "python-igraph", "pre-commit", "black==20.8b1"],
        "jax": ["jax"],
        "mpi": ["mpi4py>=3.0.1"],
    },
)
