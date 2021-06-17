from netket.utils.group.planar import C, D
from netket.graph import Triangular
from timeit import timeit


for size in [5,10,15,20]:
    lattice = Triangular((size,size))
    print(f'{size} x {size} lattice')
    for pg,pgn in [(C(3),'C(3)'), (C(6),'C(6)'), (D(3),'D(3)'), (D(6),'D(6)')]:
        print(f'* {pgn}')
        g = lattice.space_group(pg)
        t = timeit(lambda: g._irrep_matrices, number=1)
        print(f'  * New implementation: {t:.6f} sec')
        # need to reset caches
        g = lattice.space_group(pg)
        t = timeit(lambda: g.old_irrep_matrices, number=1)
        print(f'  * Old implementation: {t:.6f} sec')
    
