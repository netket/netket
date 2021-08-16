import netket as nk
from netket.optimizer import qgt
from flax.core import unfreeze
from flax import linen as nn
from flax.linen.initializers import normal, variance_scaling
import jax
from jax import numpy as jnp
from jax.scipy.sparse.linalg import cg
import numpy as np
from typing import Any
from netket.utils.types import NNInitFunc
import timeit

from sys import argv


def timeit_gc(x, number=1):
    return timeit.Timer(x, "gc.enable()").timeit(number=number)


# Return the proper kind of QGT
def QGT(*args, **kwargs):
    if argv[1] == "pytree":
        return qgt.QGTJacobianPyTree(*args, **kwargs)
    elif argv[1] == "onthefly":
        return qgt.QGTOnTheFly(*args, **kwargs)
    else:
        raise Exception("bad kind of QGT")


def construct_and_solve(vstate, rhs):
    qgt_ = QGT(vstate=vstate, diag_shift=0.01)
    return jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs))


# Benchmark starts here
def benchmark(side, n_samples, layers, features):
    n_nodes = side * side
    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    graph = nk.graph.Square(side)
    hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)

    machine = nk.models.GCNN(
        symmetries=graph, features=features, layers=layers, dtype=float, mode=argv[2]
    )

    # Create a variational state to run QGT on
    sa = nk.sampler.MetropolisExchange(
        hilbert=hilbert, graph=graph, d_max=2, n_chains=n_samples
    )
    vstate = nk.variational.MCState(
        sampler=sa, model=machine, n_samples=n_samples, n_discard_per_chain=2
    )
    vstate.init(seed=0)
    Tsamp_1 = timeit_gc(lambda: vstate.samples.block_until_ready())

    # Generate a random RHS of the same pytree shape as the parameters
    vec, unravel = nk.jax.tree_ravel(vstate.parameters)
    vec1 = jax.random.normal(keys[1], shape=vec.shape, dtype=vec.dtype)
    rhs1 = unravel(vec1)
    vec2 = jax.random.normal(keys[2], shape=vec.shape, dtype=vec.dtype)
    rhs2 = unravel(vec2)
    # Generate a new set of parameters for the second set of runs
    vecp = vec * jax.random.normal(keys[3], shape=vec.shape, dtype=vec.dtype)
    pars = unravel(vecp)

    Tjit = timeit_gc(
        lambda: construct_and_solve(vstate, rhs1),
        number=1,
    )

    # See what jit hath wrought us
    vstate._parameters = pars
    Tsamp_2 = timeit_gc(lambda: vstate.samples.block_until_ready())
    Tsamp = (Tsamp_1 + Tsamp_2) / 2

    # Time of full SR cycle
    Tcycle = (
        timeit_gc(
            lambda: construct_and_solve(vstate, rhs2),
            number=5,
        )
        / 5
    )

    # Time of solving only
    qgt_ = QGT(vstate=vstate, diag_shift=0.01)

    Tsolve = (
        timeit_gc(
            lambda: jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs2)),
            number=5,
        )
        / 5
    )

    print(
        f"{side}\t{features}\t{layers}\t{n_samples}\t{Tsamp:.6f}\t{Tjit:.6f}\t{Tcycle:.6f}\t{Tsolve:.6f}"
    )


print(f"# GCNN mode {argv[2]}")
print(f"# Side length\tFeatures\tLayers\tSamples\tSampling\tJitting\tSR cycle\tSolve")

# Different system sizes
benchmark(6, 256, 4, 4)
benchmark(8, 256, 4, 4)
benchmark(10, 256, 4, 4)
benchmark(12, 256, 4, 4)

# Different feature count
benchmark(6, 256, 4, 2)
benchmark(6, 256, 4, 4)
benchmark(6, 256, 4, 6)
benchmark(6, 256, 4, 8)

# Different number of layers
benchmark(6, 256, 4, 4)
benchmark(6, 256, 6, 4)
benchmark(6, 256, 8, 4)
benchmark(6, 256, 10, 4)

# Different sample numbers
benchmark(6, 128, 4, 4)
benchmark(6, 256, 4, 4)
benchmark(6, 384, 4, 4)
benchmark(6, 512, 4, 4)
benchmark(6, 768, 4, 4)
benchmark(6, 1024, 4, 4)
