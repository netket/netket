from netket.graph import Pyrochlore
from netket.utils.group.cubic import T,Td,Fd3m
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

for size in [2,3,4]:
    lattice = Pyrochlore((size,size,size))
    print(f'{size} x {size} x {size} primitive pyrochlore cells')
    for pg,pgn in [(T(),'T'), (Td(),'Td'), (Fd3m(),'Fd3m')]:
        print(f'* {pgn}')
        benchmark(lambda: lattice.space_group(pg))
