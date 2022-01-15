from netket.nn import DenseSymm, DenseEquivariant
from netket.graph import Square
import numpy as np
import jax
from timeit import timeit

# Test DenseSymm

for L in [2, 4, 6, 8, 10]:
    dum_input = np.random.normal(0, 1, [1000, L * L])
    graph = Square(L)
    ma_fft = DenseSymm(symmetries=graph, mode="fft", features=4)
    ma_matrix = DenseSymm(symmetries=graph, mode="matrix", features=4)

    params_fft = ma_fft.init(jax.random.PRNGKey(0), dum_input)
    apply_fft = jax.jit(lambda x: ma_fft.apply(params_fft, x))

    params_matrix = ma_matrix.init(jax.random.PRNGKey(0), dum_input)
    apply_matrix = jax.jit(lambda x: ma_matrix.apply(params_matrix, x))

    print("timing DenseSymm fft mode, L = " + str(L))
    compile_time = timeit(lambda: apply_fft(dum_input).block_until_ready(), number=1)
    runtime = timeit(lambda: apply_fft(dum_input).block_until_ready(), number=10) / 10
    print("compile time = " + str(compile_time) + " runtime is " + str(runtime))

    print("timing DenseSymm matrix mode, L = " + str(L))
    compile_time = timeit(lambda: apply_matrix(dum_input).block_until_ready(), number=1)
    runtime = (
        timeit(lambda: apply_matrix(dum_input).block_until_ready(), number=10) / 10
    )
    print("compile time = " + str(compile_time) + " runtime is " + str(runtime))


# Test DenseEquivariant

for L in [2, 4, 6, 8, 10]:

    graph = Square(L)
    sg = graph.space_group()
    irreps = sg.irrep_matrices()
    product_table = sg.product_table

    ma_fft = DenseEquivariant(
        symmetries=product_table,
        mode="fft",
        shape=graph.extent,
        features=4,
    )
    ma_irreps = DenseEquivariant(symmetries=irreps, mode="irreps", features=4)
    ma_matrix = DenseEquivariant(symmetries=product_table, mode="matrix", features=4)

    dum_input = np.random.normal(0, 1, [1000, 4, len(sg)])

    params_fft = ma_fft.init(jax.random.PRNGKey(0), dum_input)
    apply_fft = jax.jit(lambda x: ma_fft.apply(params_fft, x))

    params_irreps = ma_irreps.init(jax.random.PRNGKey(0), dum_input)
    apply_irreps = jax.jit(lambda x: ma_irreps.apply(params_irreps, x))

    params_matrix = ma_matrix.init(jax.random.PRNGKey(0), dum_input)
    apply_matrix = jax.jit(lambda x: ma_matrix.apply(params_matrix, x))

    print("timing DenseEquivariant fft mode, L = " + str(L))
    compile_time = timeit(lambda: apply_fft(dum_input).block_until_ready(), number=1)
    runtime = timeit(lambda: apply_fft(dum_input).block_until_ready(), number=10) / 10
    print("compile time = " + str(compile_time) + " runtime is " + str(runtime))

    print("timing DenseEquivariant irreps mode, L = " + str(L))
    compile_time = timeit(lambda: apply_irreps(dum_input).block_until_ready(), number=1)
    runtime = (
        timeit(lambda: apply_irreps(dum_input).block_until_ready(), number=10) / 10
    )
    print("compile time = " + str(compile_time) + " runtime is " + str(runtime))

    print("timing DenseEquivariant matrix mode, L = " + str(L))
    compile_time = timeit(lambda: apply_matrix(dum_input).block_until_ready(), number=1)
    runtime = (
        timeit(lambda: apply_matrix(dum_input).block_until_ready(), number=10) / 10
    )
    print("compile time = " + str(compile_time) + " runtime is " + str(runtime))
