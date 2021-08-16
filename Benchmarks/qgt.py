import netket as nk
from netket.optimizer import qgt
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


class FFNN(nn.Module):
    n_layers: int
    width: int
    dtype: Any = np.float64
    activation: Any = jax.nn.selu
    kernel_init: NNInitFunc = variance_scaling(1.0, "fan_in", "normal")
    bias_init: NNInitFunc = normal(0.01)

    def setup(self):
        self.layers = [
            nk.nn.Dense(
                features=self.width,
                use_bias=True,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                bias_init=self.bias_init,
            )
            for layer in range(self.n_layers)
        ]

    @nn.compact
    def __call__(self, x_in):
        x = x_in
        for layer in self.layers:
            x = layer(x)
            x = self.activation(x)
        return jnp.sum(x, axis=-1) / (x.shape[-1]) ** 0.5


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
def _benchmark(n_nodes, n_samples, n_layers, width):
    keys = jax.random.split(jax.random.PRNGKey(0), 3)

    graph = nk.graph.Chain(n_nodes)
    hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
    machine = FFNN(n_layers=n_layers, width=width)

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
    vec1 = jax.random.normal(keys[0], shape=vec.shape, dtype=vec.dtype)
    rhs1 = unravel(vec1)
    vec2 = jax.random.normal(keys[1], shape=vec.shape, dtype=vec.dtype)
    rhs2 = unravel(vec2)
    # Generate a new set of parameters for the second set of runs
    vecp = vec * jax.random.normal(keys[2], shape=vec.shape, dtype=vec.dtype)
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
        f"{n_nodes}\t{n_layers}\t{n_samples}\t{Tsamp:.6f}\t{Tjit:.6f}\t{Tcycle:.6f}\t{Tsolve:.6f}"
    )


def benchmark(n_nodes, n_samples, n_layers, width, pytree_safe=True):
    if argv[1] != "pytree" or pytree_safe:
        _benchmark(n_nodes, n_samples, n_layers, width)


print(f"# Nodes\tLayers\tSamples\tSampling\tJitting\tSR cycle\tSolve")

# Different network widths/system sizes
benchmark(256, 256, 4, 256)
benchmark(512, 256, 4, 512)
benchmark(1024, 256, 4, 1024)
benchmark(2048, 256, 4, 2048, False)
benchmark(4096, 256, 4, 4096, False)

# Different number of layers
benchmark(512, 256, 4, 512)
benchmark(512, 256, 8, 512)
benchmark(512, 256, 16, 512)
benchmark(512, 256, 32, 512, False)

# Different sample numbers
benchmark(512, 256, 4, 512)
benchmark(512, 512, 4, 512)
benchmark(512, 1024, 4, 512)
benchmark(512, 2048, 4, 512, False)
benchmark(512, 4096, 4, 512, False)
