from netket.utils.group.planar import C, D
from netket.graph import Triangular, Square
from timeit import timeit
import numpy as np

def benchmark(gen):
    g = gen()
    t1 = timeit(lambda: g._irrep_matrices, number=1)
    g = gen()
    t2 = timeit(lambda: g.old_irrep_matrices, number=1)
    
    squares = np.diag(g.product_table[g.inverse])
    frob = np.array(np.rint(np.sum(g.character_table()[:,squares], axis=1)/len(g)), dtype=int)

    real = np.sum(frob == 1)
    cplx = np.sum(frob == 0)
    quat = np.sum(frob == -1)

    print(f'  * {real} real, {cplx} complex, {quat} quaternionic irreps')
    print(f'  * New implementation: {t1:.6f} sec')
    print(f'  * Old implementation: {t2:.6f} sec')

for size in [4,6,9,12]:
    lattice = Triangular((size,size))
    print(f'{size} x {size} triangular lattice')
    for pg,pgn in [(C(3),'C(3)'), (C(6),'C(6)'), (D(3),'D(3)'), (D(6),'D(6)')]:
        print(f'* {pgn}')
        benchmark(lambda: lattice.space_group(pg))

for size in [3,6,9,12]:
    lattice = Square(size)
    print(f'{size} x {size} square lattice')
    for pg,pgn in [(C(2), 'C(2)'), (C(4), 'C(4)'), (D(2), 'D(2)'), (D(4), 'D(4)')]:
        print(f'* {pgn}')
        benchmark(lambda: lattice.space_group(pg))
