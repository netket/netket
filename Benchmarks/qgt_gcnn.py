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


def timeit_gc(x, number=1):
    return timeit.Timer(x, "gc.enable()").timeit(number=number)


# Benchmark starts here
def benchmark(side, n_samples, layers, features, pure_jax=True):
    n_nodes = side * side
    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    graph = nk.graph.Square(side)
    hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
    symm_group = graph.automorphisms()

    if pure_jax:
        symm = nk.utils.HashableArray(np.asarray(symm_group))
        pt = nk.utils.HashableArray(np.asarray(symm_group.product_table.ravel()))
        machine = nk.models.GCNN(
            symmetries=symm,
            flattened_product_table=pt,
            layers=layers,
            features=features,
        )
    else:
        machine = nk.models.GCNN(
            symmetries=symm_group, layers=layers, features=features
        )
    pure_jax = 1 if pure_jax else 0

    # Create a variational state to run QGT on
    sa = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph, d_max=2)
    vstate = nk.variational.MCState(sampler=sa, model=machine, n_samples=n_samples)
    vstate.init(seed=0)
    # We don't actually want to perform a rather slow sampling
    vstate._samples = hilbert.random_state(key=keys[0], size=n_samples)

    qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01)

    # Generate a random RHS of the same pytree shape as the parameters
    vec, unravel = nk.jax.tree_ravel(vstate.parameters)
    vec1 = jax.random.normal(keys[1], shape=vec.shape, dtype=vec.dtype)
    rhs1 = unravel(vec1)
    vec2 = jax.random.normal(keys[2], shape=vec.shape, dtype=vec.dtype)
    rhs2 = unravel(vec2)
    # Generate a new set of parameters for the second set of runs
    vecp = vec * jax.random.normal(keys[3], shape=vec.shape, dtype=vec.dtype)
    pars = unravel(vecp)

    time1 = timeit_gc(
        lambda: jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs1)),
        number=1,
    )

    # See what jit hath wrought us
    vstate._samples = hilbert.random_state(key=keys[4], size=n_samples)
    vstate._parameters = pars
    qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01)

    time2 = (
        timeit_gc(
            lambda: jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs2)),
            number=5,
        )
        / 5
    )
    print(
        f"{side}\t{features}\t{layers}\t{n_samples}\t{pure_jax}\t{time1:.6f}\t{time2:.6f}"
    )


print(f"# Side length\tFeatures\tLayers\tSamples\tPure JAX\tJitting\tAfter jitting")

# Different system sizes
benchmark(6, 256, 4, 4)
benchmark(8, 256, 4, 4)
benchmark(12, 256, 4, 4)
# benchmark(16, 256, 4, 4)

# Different feature count
benchmark(8, 256, 4, 2)
benchmark(8, 256, 4, 4)
benchmark(8, 256, 4, 6)
# benchmark(8, 256, 4, 8)
# benchmark(8, 256, 4, 12)

# Different sample numbers
benchmark(8, 256, 4, 4)
benchmark(8, 512, 4, 4)
benchmark(8, 1024, 4, 4)
benchmark(8, 2048, 4, 4)

# Different number of layers
benchmark(8, 256, 4, 4)
benchmark(8, 256, 8, 4)
benchmark(8, 256, 12, 4)
# benchmark(8, 256, 32, 4)

# Pure JAX?
benchmark(8, 256, 4, 4, False)
