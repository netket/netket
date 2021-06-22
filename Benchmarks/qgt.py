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
from timeit import timeit


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


# Benchmark starts here
def benchmark(n_nodes, n_samples, n_layers, width):
    keys = jax.random.split(jax.random.PRNGKey(0), 5)

    graph = nk.graph.Chain(n_nodes)
    hilbert = nk.hilbert.Spin(s=0.5, N=n_nodes)
    machine = FFNN(n_layers=n_layers, width=width)

    # Create a variational state to run QGT on
    sa = nk.sampler.MetropolisExchange(hilbert=hilbert, graph=graph, d_max=2)
    vstate = nk.variational.MCState(sampler=sa, model=machine, n_samples=n_samples)
    vstate.init(seed=0)
    # We don't actually want to perform a rather slow sampling
    vstate._samples = hilbert.random_state(key=keys[0], size=n_samples)

    qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01, centered=False)

    # Generate a random RHS of the same pytree shape as the parameters
    vec, unravel = nk.jax.tree_ravel(vstate.parameters)
    vec1 = jax.random.normal(keys[1], shape=vec.shape, dtype=vec.dtype)
    rhs1 = unravel(vec1)
    vec2 = jax.random.normal(keys[2], shape=vec.shape, dtype=vec.dtype)
    rhs2 = unravel(vec2)
    # Generate a new set of parameters for the second set of runs
    vecp = vec * jax.random.normal(keys[3], shape=vec.shape, dtype=vec.dtype)
    pars = unravel(vecp)

    time1 = timeit(lambda: jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs1)), number=1)

    # See what jit hath wrought us
    vstate._samples = hilbert.random_state(key=keys[4], size=n_samples)
    vstate._parameters = pars
    qgt_ = qgt.QGTOnTheFly(vstate=vstate, diag_shift=0.01, centered=False)

    time2 = timeit(lambda: jax.tree_map(lambda x: x.block_until_ready(), qgt_.solve(cg, rhs2)), number = 10)/10
    print(f'{n_nodes}\t{width}\t{n_layers}\t{n_samples}\t{time1:.6f}\t{time2:.6f}')

print(f'# Nodes\tWidth\tLayers\tSamples\tJitting\tAfter jitting')

# Different network widths/system sizes
benchmark(256, 256, 4, 256)
benchmark(512, 256, 4, 512)
benchmark(1024, 256, 4, 1024)
benchmark(2048, 256, 4, 2048)
benchmark(4096, 256, 4, 4096)

# Different sample numbers
benchmark(512, 256, 4, 512)
benchmark(512, 512, 4, 512)
benchmark(512, 1024, 4, 512)
benchmark(512, 2048, 4, 512)
benchmark(512, 4096, 4, 512)

# Different number of layers
benchmark(512, 256, 4, 512)
benchmark(512, 256, 8, 512)
benchmark(512, 256, 16, 512)
benchmark(512, 256, 32, 512)
