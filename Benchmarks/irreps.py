from netket.utils.group.planar import C, D
from netket.utils.group.cubic import T, Td, Fd3m
from netket.graph import Triangular, Square, Pyrochlore
from timeit import timeit
import numpy as np


def benchmark(gen):
    g = gen()
    t0 = timeit(lambda: g.product_table, number=1)
    tx = timeit(lambda: g.character_table(), number=1)
    t1 = timeit(lambda: g._irrep_matrices, number=1)
    # g = gen()
    # t2 = timeit(lambda: g.old_irrep_matrices, number=1)

    squares = np.diag(g.product_table[g.inverse])
    frob = np.array(
        np.rint(np.sum(g.character_table()[:, squares], axis=1) / len(g)), dtype=int
    )
    dims = np.array(np.rint(g.character_table()[:, 0]), dtype=int)

    real = np.sum(frob == 1)
    cplx = np.sum(frob == 0)
    quat = np.sum(frob == -1)
    label = {1: "(R)", 0: "(C)", -1: "(Q)"}

    print(f"  * Group size: {len(g)}")
    print(f"  * {real} real, {cplx} complex, {quat} quaternionic irreps")
    print(
        "  * Irrep dimensions: "
        + ", ".join([f"{dims[i]}{label[frob[i]]}" for i in range(len(frob))])
    )
    print(f"  * New implementation: {t1:.6f} sec")
    print(f"  * Calculating the product table: {t0:.6f} sec")
    print(f"  * Calculating the character table: {tx:.6f} sec")
    # print(f'  * Old implementation: {t2:.6f} sec')


for size in [4, 6, 9, 12]:
    lattice = Triangular((size, size))
    print(f"{size} x {size} triangular lattice")
    for pg, pgn in [(C(3), "C(3)"), (C(6), "C(6)"), (D(3), "D(3)"), (D(6), "D(6)")]:
        print(f"* {pgn}")
        benchmark(lambda: lattice.space_group(pg))
    print("")

for size in [3, 6, 9, 12]:
    lattice = Square(size)
    print(f"{size} x {size} square lattice")
    for pg, pgn in [(C(2), "C(2)"), (C(4), "C(4)"), (D(2), "D(2)"), (D(4), "D(4)")]:
        print(f"* {pgn}")
        benchmark(lambda: lattice.space_group(pg))
    print("")

for size in [2, 3, 4]:
    lattice = Pyrochlore((size, size, size))
    print(f"{size} x {size} x {size} primitive pyrochlore cells")
    for pg, pgn in [(T(), "T"), (Td(), "Td"), (Fd3m(), "Fd3m")]:
        print(f"* {pgn}")
        benchmark(lambda: lattice.space_group(pg))
    print("")
